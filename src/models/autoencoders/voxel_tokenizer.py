import sys
import copy
from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional, Tuple, Type, Union

import torch.utils.checkpoint as checkpoint
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKLOutput, DecoderOutput
from diffusers.models.modeling_utils import ModelMixin
from torch import Tensor, nn
from torch.nn import Sequential
import numpy as np

from ...configs.base_config import InstantiateConfig
from ...models.autoencoders.components import *
from ...utils import load_model, log_to_rank0

import torch.nn.functional as F
import kornia as K
from torchvision import transforms

from einops import rearrange, repeat

import trellis.models as models
from trellis.modules.transformer import AbsolutePositionEmbedder, TransformerBlock, TransformerPureCrossBlock

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
class VoxelTokenizerConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: VoxelTokenizer)
    """Target class to instantiate."""
    resolution: int = 64
    enc_pretrained: str = 'JeffreyXiang/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16'
    dec_pretrained: str = 'JeffreyXiang/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16'
    torch_hub_dir: str = '~/.cache/torch_hub/hub'
    vae_ckpt_path: Optional[str] = None
    temporal: bool = False
    num_frames: int = 77
    def from_pretrained(self, **kwargs) -> Any:
        """Returns the instantiated object using the config."""
        vae = VoxelTokenizer(self)
        vae._init_image_cond_model("dinov2_vitl14_reg")

        if self.vae_ckpt_path is not None:
            print(f"=====initialize weights from {self.vae_ckpt_path}===")
            rename_func = None
            vae = load_model(vae, self.vae_ckpt_path, rename_func=rename_func)
        
        return vae

# encoder = models.from_pretrained(opt.enc_pretrained).eval().cuda()
class VoxelTokenizer(ModelMixin):
    def __init__(
        self,
        config: VoxelTokenizerConfig,
    ):
        super(VoxelTokenizer, self).__init__()
        self.config = config
        self.temporal = config.temporal
        self.debug = True
        self.encoder = models.from_pretrained(self.config.enc_pretrained).eval().cuda()
        self.decoder = models.from_pretrained(self.config.dec_pretrained).eval().cuda()
        self.torch_hub_dir = config.torch_hub_dir
        self.num_frames = config.num_frames
        self.use_checkpoint = True
        if self.temporal:
            model_channels = 256
            out_channels = 8
            num_blocks = 2
            all_res = [self.num_frames, 16, 16, 16]
            self.input_layer = nn.Linear(8, model_channels)
            self.decode_input_layer = nn.Linear(8, model_channels)

            # temporal positional encoding
            pos_embedder = AbsolutePositionEmbedder(model_channels, 4) # have temporal
            coords = torch.meshgrid(*[torch.arange(res, device=self.device) for res in all_res], indexing='ij')
            coords = torch.stack(coords, dim=-1).reshape(-1, 4)
            temporal_pos_emb = pos_embedder(coords)
            self.register_buffer("temporal_pos_emb", temporal_pos_emb)
            
            self.decode_q_linear = PixArtAlphaCondProjection(model_channels, model_channels)

            self.blocks = nn.ModuleList([
                TransformerBlock(
                    model_channels,
                    num_heads=8,
                    mlp_ratio=4,
                    attn_mode='full',
                    use_checkpoint=self.use_checkpoint,
                    use_rope=False,
                    qk_rms_norm=True,
                )
                for i in range(num_blocks)
            ])
            self.encode_ca_block = TransformerPureCrossBlock(
                model_channels,
                model_channels,
                num_heads=8,
                mlp_ratio=4,
                use_checkpoint=self.use_checkpoint,
                qk_rms_norm_cross=True,
            )
            self.decode_ca_block = TransformerPureCrossBlock(
                model_channels,
                model_channels,
                num_heads=8,
                mlp_ratio=4,
                use_checkpoint=self.use_checkpoint,
                qk_rms_norm_cross=True,
            )
            self.act_layer = nn.SiLU()
            if not self.debug:
                self.out_layer = nn.Linear(model_channels, out_channels*2)
                self.out_layer_decoder = nn.Linear(model_channels, out_channels)
            else:
                self.out_layer = nn.Linear(out_channels*4, out_channels*2)
                self.out_layer_decoder = nn.Linear(out_channels, out_channels*4)
            
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
    
    def initialize_temporal_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.blocks.apply(_basic_init)
        self.encode_ca_block.apply(_basic_init)
        self.decode_ca_block.apply(_basic_init)

        # Zero-out output layers:
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)

        nn.init.constant_(self.out_layer_decoder.weight, 0)
        nn.init.constant_(self.out_layer_decoder.bias, 0)
    
    def encode_temporal(self, batch_voxels, sample_posterior: bool = False, return_raw: bool = False,
                        return_ss: bool = False):
        batch_size = len(batch_voxels)
        batch_latent_list = []
        batch_ss_list = []
        for i in range(batch_size):
            num_frames = len(batch_voxels[i])
            print("---", num_frames, self.num_frames)
            assert num_frames == self.num_frames
            decode_voxel_dict = {}
            encode_latent_list = []
            ss_list = []
            for frame_id in range(num_frames):
                voxels = batch_voxels[i][frame_id]
                v = voxels[:, :3]
                latent, ss = self.encode(v, return_ss=True) # 8, 16, 16, 16
                ss_list.append(ss)
                encode_latent_list.append(latent)
            
            ss = torch.cat(ss_list, dim=0) # 77, 1, 64, 64, 64
            batch_ss_list.append(ss)
            latents = torch.stack(encode_latent_list, dim=1) # 8, 77, 16, 16, 16
            batch_latent_list.append(latents)
        latents = torch.stack(batch_latent_list) # b 8 77 16 16 16
        ss = torch.stack(batch_ss_list) # b 77 1 64 64 64

        if not self.debug:
            h = rearrange(latents, 'b c f h w d -> b (f h w d) c')
            h = self.input_layer(h)
            h = h + self.temporal_pos_emb[None]
            # sa
            for block in self.blocks:
                h = block(h)
            # ca
            h = rearrange(h, 'b (f n) c -> b f n c', f=self.num_frames)
            h = rearrange(h, 'b (f k) n c -> b f k n c', k=4)
            q = h[:, :, -1] # b f//4 n c
            q = rearrange(q, 'b f n c -> (b f) n c')
            h = rearrange(h, 'b f k n c -> (b f) (k n) c')
            h = self.encode_ca_block(q, h)
            h = F.layer_norm(h, h.shape[-1:])
            # h = self.act_layer(h)
            h = self.out_layer(h)
            h = rearrange(h, '(b f) n c -> b f n c', b=batch_size)
        else:
            print("===debug encoder===")
            h = rearrange(latents, 'b c (f k) h w d -> b f (h w d) (c k)', k=4)
            h = self.out_layer(h) # b n c
        
        mean, logvar = h.chunk(2, dim=-1)

        if sample_posterior:
            std = torch.exp(0.5 * logvar)
            z = mean + std * torch.randn_like(std)
        else:
            z = mean
        
        if return_raw:
            if return_ss:
                return z, mean, logvar, ss
            return z, mean, logvar
        if return_ss:
            return z, ss
        return z

    
    def deocode_temporal(self, latent):
        # latent b f (hwd) c
        if not self.debug:
            latent = rearrange(latent, 'b f n c -> b (f n) c')
            latent = self.decode_input_layer(latent)
            q = self.decode_q_linear(self.temporal_pos_emb[None]) # 1, n c [n=fhwd]
            q = repeat(q, "1 n d -> b n d", b=latent.shape[0]) # b n c
            h = self.decode_ca_block(q, latent) # b (fhwd) c
            h = F.layer_norm(h, h.shape[-1:])
            h = self.out_layer_decoder(h)
            h = rearrange(h, 'b (f h w d) c -> b f c h w d', f=self.num_frames, h=16, w=16, d=16)
        else:
            # b f n c
            print("===debug decoder===")
            h = self.out_layer_decoder(latent)
            h = rearrange(h, "b f (h w d) (c k) -> b (f k) c h w d", k=4, h=16, w=16, d=16)
        # run original decoder
        batch_size = h.shape[0]
        all_decode_latents = []
        for i in range(batch_size):
            latent = h[i]
            decode_latent_list = []
            for frame_id in range(self.num_frames):
                logits = self.decoder(latent[frame_id][None]) # 1, 8, 16, 16, 16
                # print('===logits===', logits.shape) # 1 1 64 64 64
                decode_latent_list.append(logits)
            decode_latents = torch.cat(decode_latent_list, dim=0)
            all_decode_latents.append(decode_latents)
        decode_latents = torch.stack(all_decode_latents) # b f 1 64 64 64
        return decode_latents

    def encode(self, position, return_ss=False):
        # position: tensor, bs=1
        ss = self.get_voxels(position).cuda().float()[None]
        # ss = ss.to(self.encoder.dtype) 
        ss = ss.to(self.encoder.input_layer.weight.dtype) 
        latent = self.encoder(ss, sample_posterior=False)
        assert torch.isfinite(latent).all(), "Non-finite latent"
        if return_ss:
            return latent[0], ss.to(self.encoder.dtype)
        return latent[0]
    
    def decode(self, latent):
        # Decode occupancy latent
        coords = torch.argwhere(self.decoder(latent)>0)[:, [0, 2, 3, 4]].int() # b, 1, h, w, d -> b h w d
        return coords
    
    @torch.no_grad()
    def encode_cond_img(self, image=None):
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            from PIL import Image
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
        features = self.image_cond_model(image.float(), is_training=True)['x_prenorm']
        patchtokens = F.layer_norm(features, features.shape[-1:])
        patchtokens = patchtokens.to(image.dtype)
        return patchtokens
    
    def _init_image_cond_model(self, name: str):
        """
        Initialize the image conditioning model.
        """
        import os
        self.torch_hub_dir = os.path.expanduser(self.torch_hub_dir)
        os.environ["TORCH_HOME"] = os.path.dirname(self.torch_hub_dir)
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