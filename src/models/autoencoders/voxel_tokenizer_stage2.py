import sys
import copy
from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional, Tuple, Type, Union

import torch.utils.checkpoint as checkpoint
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKLOutput, DecoderOutput
from diffusers.models.modeling_utils import ModelMixin
from torch import Tensor, nn
from torch.nn import Sequential
import cv2
import numpy as np

from ...configs.base_config import InstantiateConfig
from ...models.autoencoders.components import *
from ...utils import load_model, log_to_rank0

import torch.nn.functional as F
import kornia as K
from torchvision import transforms

import trellis.models as models
import trellis.modules.sparse as sp

class PixArtAlphaCondProjection(nn.Module):
    """
    Projects caption embeddings. Also handles dropout for classifier-free guidance.

    Adapted from https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
    """

    def __init__(self, in_features, hidden_size, out_features=None, act_fn="gelu_tanh"):
        super().__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear_1 = nn.Linear(in_features=in_features, out_features=hidden_size, bias=True)
        if act_fn == "gelu_tanh":
            self.act_1 = nn.GELU(approximate="tanh")
        elif act_fn == "silu":
            self.act_1 = nn.SiLU()
        elif act_fn == "silu_fp32":
            raise NotImplementedError
            # self.act_1 = FP32SiLU()
        else:
            raise ValueError(f"Unknown activation function: {act_fn}")
        self.linear_2 = nn.Linear(in_features=hidden_size, out_features=out_features, bias=True)

    def forward(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states

@dataclass
class VoxelTokenizerConfigStage2(InstantiateConfig):
    _target: Type = field(default_factory=lambda: VoxelTokenizerStage2)
    """Target class to instantiate."""
    # resolution: int = 64
    enc_pretrained: str = 'JeffreyXiang/TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16'
    dec_pretrained: str = 'JeffreyXiang/TRELLIS-image-large/ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16'
    dec_gs_pretrained: str = 'JeffreyXiang/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16'
    torch_hub_dir: str = '~/.cache/torch_hub/hub'
    vae_ckpt_path: Optional[str] = None
    temporal: bool = False
    num_frames: int = 77
    def from_pretrained(self, **kwargs) -> Any:
        """Returns the instantiated object using the config."""
        vae = VoxelTokenizerStage2(self)
        vae._init_image_cond_model("dinov2_vitl14_reg")

        if self.vae_ckpt_path is not None:
            print(f"=====initialize weights from {self.vae_ckpt_path}===")
            rename_func = None
            vae = load_model(vae, self.vae_ckpt_path, rename_func=rename_func)
        
        return vae

# encoder = models.from_pretrained(opt.enc_pretrained).eval().cuda()
class VoxelTokenizerStage2(ModelMixin):
    def __init__(
        self,
        config: VoxelTokenizerConfigStage2,
    ):
        super(VoxelTokenizerStage2, self).__init__()
        self.config = config
        self.temporal = config.temporal
        self.debug = True
        self.encoder = models.from_pretrained(self.config.enc_pretrained).eval().cuda()
        self.decoder = models.from_pretrained(self.config.dec_pretrained).eval().cuda()
        self.decoder_gs = models.from_pretrained(self.config.dec_gs_pretrained).eval().cuda()
        self.torch_hub_dir = config.torch_hub_dir
        self.num_frames = config.num_frames
        self.use_checkpoint = True
        
        # self.initialize_temporal_weights()

    
    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.encoder.convert_to_fp16()
        self.decoder.convert_to_fp16()
    
    def convert_to_bf16(self) -> None:
        """
        Convert the torso of the model to bfloat16.
        """
        self.encoder.convert_to_bf16()
        self.decoder.convert_to_bf16()

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.encoder.convert_to_fp32()
        self.decoder.convert_to_fp32()
    
    def get_voxels(self, position):
        coords = ((torch.tensor(position) + 0.5) * self.config.resolution).int().contiguous()
        ss = torch.zeros(1, self.config.resolution, self.config.resolution, self.config.resolution, dtype=torch.long)
        ss[:, coords[:, 0], coords[:, 1], coords[:, 2]] = 1
        return ss

    def encode(self, feats, return_ss=False):

        feats = sp.SparseTensor(
                    feats = torch.from_numpy(feats['patchtokens']).float(),
                    coords = torch.cat([
                        torch.zeros(feats['patchtokens'].shape[0], 1).int(),
                        torch.from_numpy(feats['indices']).int(),
                    ], dim=1),
                ).cuda()
        latent = self.encoder(feats, sample_posterior=False)
        assert torch.isfinite(latent.feats).all(), "Non-finite latent"
        # pack = {
        #     'feats': latent.feats.cpu().numpy().astype(np.float32),
        #     'coords': latent.coords[:, 1:].cpu().numpy().astype(np.uint8),
        # }
        return latent
    
    def decode(
        self,
        slat: sp.SparseTensor,
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
    ) -> dict:
        """
        Decode the structured latent.

        Args:
            slat (sp.SparseTensor): The structured latent.
            formats (List[str]): The formats to decode the structured latent to.

        Returns:
            dict: The decoded structured latent.
        """
        ret = {}
        if 'mesh' in formats:
            ret['mesh'] = self.decoder(slat)
        if 'gaussian' in formats:
            ret['gaussian'] = self.decoder_gs(slat)
        if 'radiance_field' in formats:
            raise NotImplementedError
            # ret['radiance_field'] = self.decoder(slat)
        return ret
    
    @torch.no_grad()
    def encode_cond_img(self, image=None):
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
            image = [i.resize((518, 518), Image.LANCZOS) for i in image]
            image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).to('cuda')
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")
        image = self.image_cond_model_transform(image).to('cuda')
        features = self.image_cond_model(image, is_training=True)['x_prenorm']
        patchtokens = F.layer_norm(features, features.shape[-1:])
        return patchtokens
    
    @torch.no_grad()
    def encode_cond_img_from_tensor(self, image):
        # print('===into cond model img===', image.dtype, image.shape)
        features = self.image_cond_model(image.float(), is_training=True)['x_prenorm']
        patchtokens = F.layer_norm(features, features.shape[-1:])
        patchtokens = patchtokens.to(image.dtype)
        # print("====enc img cond====", patchtokens.shape, patchtokens.dtype)
        return patchtokens
    
    def _init_image_cond_model(self, name: str):
        """
        Initialize the image conditioning model.
        """
        import os
        os.environ["TORCH_HUB_DIR"] = self.torch_hub_dir
        repo_dir = f"{self.torch_hub_dir}/facebookresearch_dinov2_main"

        dinov2_model = torch.hub.load(
            repo_or_dir=repo_dir, 
            model=name, 
            source='local',
            pretrained=True
        )
        print("=====================dino v2 loaded====================")
        dinov2_model.eval().cuda()
        self.image_cond_model = dinov2_model
        transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.image_cond_model_transform = transform