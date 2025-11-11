import math
from dataclasses import dataclass, field
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import torch
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
from einops import rearrange, repeat
import numpy as np

# from TRELLIS.trellis.modules import attention
from ..configs.config_utils import to_immutable_dict
from ..models import VisualTokenizerConfig, VoxelTokenizerConfig, VoxelTokenizer
from ..models.transformers import Transformer2DModelConfig
from ..utils import measure_time
from .base_pipeline import EDMTrainConfig, PipelineMixin
from .t2v_pipeline_pixart_alpha import T2VPixArtAlphaPipeline, T2VPixArtAlphaPipelineConfig
from ..utils import pack_data, pack_data_voxel
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer
from diffusers.models import AutoencoderKL, Transformer2DModel
from diffusers.schedulers import DPMSolverMultistepScheduler
from torchvision.utils import save_image

import trellis.modules.sparse as sp
from trellis.utils import render_utils, postprocessing_utils

from PIL import Image

from torchvision import transforms
import open3d as o3d

def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    denormalized = tensor.clone()
    for c in range(3):
        denormalized[:, c] = denormalized[:, c] * std[c] + mean[c]
    return denormalized

def voxelize_point_cloud(points):
    clipped_points = np.clip(points, -0.5 + 1e-6, 0.5 - 1e-6)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(clipped_points)

    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
        pcd,
        voxel_size=1/64,
        min_bound=(-0.5, -0.5, -0.5),
        max_bound=(0.5, 0.5, 0.5)
    )
    voxels = voxel_grid.get_voxels()
    if len(voxels) == 0:
        raise ValueError("No voxels generated - input points may be empty")
    indices = np.array([voxel.grid_index for voxel in voxels])
    assert np.all(indices >= 0) and np.all(indices < 64), "Voxel indices out of bounds"
    
    return indices

def get_cond_imgs(path):
    image = Image.open(path)
    # assert image.mode == 'RGBA'
    print("---image mode---", image.mode, path, np.array(image).shape)
    if image.mode != 'RGBA' or np.array(image).shape[0] != np.array(image).shape[1]:
        print("---------------------preprocess image-------------------", path)
        image = preprocess_image(image)
        image = [image]
        assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
        image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
        image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
        image = torch.stack(image) # 1, 3, h, w
        return image
    image = [image]
    assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
    image = [i.resize((518, 518), Image.LANCZOS) for i in image]
    image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
    image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
    image = torch.stack(image) # 1, 3, h, w
    return image

def preprocess_image(input: Image.Image) -> Image.Image:
    """
    Preprocess the input image.
    """
    # if has alpha channel, use it directly; otherwise, remove background
    has_alpha = False
    if input.mode == 'RGBA':
        alpha = np.array(input)[:, :, 3]
        if not np.all(alpha == 255):
            has_alpha = True
    if has_alpha:
        output = input
    else:
        input = input.convert('RGB')
        max_size = max(input.size)
        scale = min(1, 1024 / max_size)
        if scale < 1:
            input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
        import rembg
        rembg_session = rembg.new_session('u2net')
        output = rembg.remove(input, session=rembg_session)
    output_np = np.array(output)
    alpha = output_np[:, :, 3]
    bbox = np.argwhere(alpha > 0.8 * 255)
    bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    size = int(size * 1.2)
    # size = int(size * 1.5)
    bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
    output = output.crop(bbox)  # type: ignore
    output = output.resize((518, 518), Image.Resampling.LANCZOS)
    output = np.array(output).astype(np.float32) / 255
    output = output[:, :, :3] * output[:, :, 3:4]
    output = Image.fromarray((output * 255).astype(np.uint8))
    return output

@dataclass
class JointDiTSingle3DPipelineConfigStage2(T2VPixArtAlphaPipelineConfig):
    """Configuration for Pipeline instantiation"""

    _target: Type = field(default_factory=lambda: JointDiTSingle3DPipelineStage2)
    """target class to instantiate"""

    edm_config: Optional[EDMTrainConfig] = None
    """EDM config, default to EDM"""

    epsilon: float = 1e-3
    epsilon_voxel: float = 1e-5

    transformer_config: Transformer2DModelConfig = Transformer2DModelConfig()
    """The transformer config for the pipeline."""
    vae_config: VisualTokenizerConfig = VisualTokenizerConfig()
    """The VAE config for the pipeline."""

    voxel_vae_config: VoxelTokenizerConfig = VoxelTokenizerConfig()
    """The VAE config for voxels"""

    proportion_empty_prompts: float = 0.1
    """Proportion of empty prompts to use."""

    max_sequence_length: int = 256
    """The maximum number of tokens for text encoding"""
    logit_normal: bool = True
    timestep_shift: float = 1.0
    match_snr: bool = False

    call: Dict[str, Any] = to_immutable_dict(
        {
            "num_frames": 16,
            "height": 256,
            "width": 384,
            "num_inference_steps": 50,
        }
    )
    """The inference call arguments for the pipeline."""
    measure_time: bool = False
    s1_save_dir: str = './sample_results'


class JointDiTSingle3DPipelineStage2(T2VPixArtAlphaPipeline, PipelineMixin):

    _optional_components = ["transformer"]
    
    def __init__(
        self,
        tokenizer: T5Tokenizer = None,
        text_encoder: T5EncoderModel = None,
        vae: AutoencoderKL = None,
        transformer: Transformer2DModel = None,
        scheduler: DPMSolverMultistepScheduler = None,
        clip_g_model = None,
        clip_l_model: CLIPTextModel = None,
        tokenizer_clip: CLIPTokenizer = None,
        voxel_vae: VoxelTokenizer = None, 
    ):
        super().__init__(tokenizer, text_encoder, vae, transformer, scheduler, clip_g_model, clip_l_model, tokenizer_clip)
        self.register_modules(voxel_vae=voxel_vae)
        self.cnt = 0
        # self.voxel_vae.convert_to_bf16()
        self.transformer.convert_to_bf16()
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls(*args, **kwargs)
        

    @torch.no_grad()
    def get_latents(self, batch_video):
        latents = [self.video2latents(sample, out_pattern="b c f h w") for sample in batch_video]
        latents, masks = pack_data(latents)
        return latents, masks

    @torch.no_grad()
    def get_latents_per_frame(self, batch_video):
        latents = []
        for sample in batch_video:
            sample_latents = []
            for frame_idx in range(sample.shape[1]):
                sample_latents.append(self.video2latents(sample[:,[frame_idx],...], out_pattern="b c f h w"))
            latents.append(torch.concatenate(sample_latents, 2))
        latents, masks = pack_data(latents)
        return latents, masks

    @torch.no_grad()
    def get_latents_voxel(self, batch_voxels):
        latents = []
        for sample in batch_voxels:
            if isinstance(sample, list):
                # sample is a dynamic sequence
                latent_list = []
                for frame_voxel in sample:
                    latent = self.voxel_vae.encode(frame_voxel) # 8, 16, 16, 16
                    latent_list.append(latent)
                latent = torch.stack(latent_list, dim=1)[None] # 1, 8, 77, 16, 16, 16
            else:
                # sample is a static voxel
                latent = self.voxel_vae.encode(sample)
                latent = latent.unsqueeze(1).unsqueeze(0)
            latents.append(latent) 
        latents, masks = pack_data_voxel(latents)
        return latents, masks

    def forward(self, batch):

        if "batch_video" in batch.keys():
            latents, attention_mask = self.get_latents(batch["batch_video"])
            batch_size = latents.shape[0]
        else:
            latents = None

        batch_part = batch['batch_part']
        batch_bbox = batch['batch_bbox']

        batch_size = len(batch_bbox)
        assert batch_size == 1
        num_parts = len(batch_bbox[0])

        batch_slat_coords = batch['batch_slat_coords']
        batch_slat_feats = batch['batch_slat_feats']
        # get img conds
        if "batch_cond_imgs" in batch.keys():
            img_conds = batch["batch_cond_imgs"]
            img_conds = torch.cat(img_conds, dim=0)
            img_conds = self.voxel_vae.encode_cond_img_from_tensor(img_conds)
        else:
            img_conds = None
        
        # drop img conds for cfg
        assert self.transformer.drop_img_conds > 0.
        import random
        if random.random() < self.transformer.drop_img_conds:
            img_conds = torch.zeros_like(img_conds)
        
        assert self.transformer.drop_global >= 0.
        if random.random() < self.transformer.drop_global:
            batch_slat_feats[0][0] = torch.zeros_like(batch_slat_feats[0][0])

        slat_coords_cat = []
        slat_feats_cat = []
        for bs_id, part_verts in enumerate(batch_part):
            # bs = 1 for now
            for i in range(len(part_verts)):
                ## decode slat
                slat_feats = batch_slat_feats[bs_id][i]
                slat_coords = batch_slat_coords[bs_id][i]
                batch_ids = torch.ones(slat_coords.shape[0], 1, dtype=slat_coords.dtype, device=slat_coords.device) * i + bs_id
                slat_coords = torch.cat([batch_ids, slat_coords], dim=-1)
                slat_coords_cat.append(slat_coords)
                slat_feats_cat.append(batch_slat_feats[bs_id][i])
        
        slat_coords_cat = torch.cat(slat_coords_cat)
        slat_feats_cat = torch.cat(slat_feats_cat)

        slat = sp.SparseTensor(
                    feats = slat_feats_cat.float(),
                    coords = slat_coords_cat,
                ).cuda()
        slat = slat.to(self.transformer.voxel_branch.input_layer.weight.dtype)
        slat = slat.to(self.transformer.voxel_branch.input_layer.weight.device)
        
        if "batch_voxels" in batch.keys():
            batch_voxels = batch["batch_voxels"]
            voxel_latents, voxel_masks = self.get_latents_voxel(batch_voxels) # b, 8, 77, 16, 16, 16
            voxel_latents = voxel_latents.to(slat.dtype)

            voxel_plucker_video_batch = batch["batch_voxel_plucker_video"] # B V N 6
            voxel_plucker_video_batch = torch.cat(voxel_plucker_video_batch, dim=0)
        
        eps_2 = self.pipeline_config.epsilon_voxel

        #NOTE: add noise for voxel latents
        # z_2 = self.generate_noise(voxel_latents)
        z_2 = slat.replace(torch.randn_like(slat.feats))

        if "t_step" in batch:
            t = batch["t_step"].repeat_interleave(batch_size).to(device=slat.device, dtype=slat.dtype)
        else:
            if self.pipeline_config.logit_normal:
                t_logit = torch.exp(torch.randn(batch_size, device=z_2.device))
                t = t_logit / (t_logit + 1)
            else:
                t = torch.rand(batch_size, device=z_2.device)

            t = self.pipeline_config.timestep_shift * t / (1 - t + self.pipeline_config.timestep_shift * t)

            timestep = t
            if self.pipeline_config.match_snr:
                raise NotImplementedError
                scale_factor = latents.shape[2] ** 0.5
                t = scale_factor * t / (1 - t + scale_factor * t)
        
        timestep = repeat(timestep, "b -> (b nv)", nv=slat.shape[0])

        # repeat cond
        img_conds = repeat(img_conds, "b n c -> (b nv) n c", nv=slat.shape[0])
        assert self.transformer.training, "not implement img cond repeat for eval now"

        #NOTE: add noise for voxel latents
        t_expand_2 = timestep.view(-1, *[1 for _ in range(len(slat.shape) - 1)])
        z_t_2 = (1 - t_expand_2) * slat + (eps_2 + (1 - eps_2) * t_expand_2) * z_2
        u_2 = (1 - eps_2) * z_2 - slat

        # process conditions
        # condition_order_list = self.pipeline_config.condition_order.split(",")

        #NOTE: add noise for voxel latents
        # z_t_2 = z_t_2.to(voxel_latents.dtype)
        # u_2 = u_2.to(voxel_latents.dtype)

        with measure_time("Transformer", self.pipeline_config.measure_time):
            v, v_2 = self.transformer(
                # z_t,
                None,
                timestep=timestep * 999,
                return_dict=False,
                noisy_voxel_latent = z_t_2,
                img_conds = img_conds,
                bboxes = batch_bbox,
                np=num_parts,
                batch_size=batch_size,
            )
        
        self.cnt += 1
        return self.compute_loss({"pred_voxel": v_2.feats, "target_voxel": u_2.feats})
    
    def compute_loss(self, result_dict):
        # mask shape = b 1 f h w
        # pred, target = result_dict["pred"], result_dict["target"]
        pred_voxel, target_voxel = result_dict["pred_voxel"], result_dict["target_voxel"]

        loss_weight_voxel = 1.
        loss_voxel = ((pred_voxel.float() - target_voxel.float()) ** 2).reshape(target_voxel.shape[0], -1).mean(dim=1)
        loss_reweight_voxel = (loss_voxel * loss_weight_voxel).mean()  # MSE loss
        loss_orginal_voxel = loss_voxel.mean()  # MSE loss

        total_loss =  loss_reweight_voxel
        return {"total_loss": total_loss, "loss_unweight_voxel": loss_orginal_voxel}

    @staticmethod
    def prepare_call_kwargs(pipeline, batch, **kwargs):
        call_kwargs = {}
        if "prompt" not in kwargs:
            # call_kwargs["prompt"] = batch["prompts"]
            call_kwargs["prompt"] = batch["batch_caption"]
        if "batch_cond_imgs" not in kwargs:
            call_kwargs["batch_cond_imgs"] = batch["batch_cond_imgs"]
        if "batch_camera_video" not in kwargs:
            call_kwargs["batch_camera_video"] = batch["batch_camera_video"]
        if "batch_voxel_plucker_video" not in kwargs:
            call_kwargs["batch_voxel_plucker_video"] = batch["batch_voxel_plucker_video"]
        if "generator" not in kwargs:
            call_kwargs["generator"] = torch.Generator(device=pipeline.device).manual_seed(pipeline.pipeline_config.seed)
        call_kwargs.update(pipeline.pipeline_config.call)
        call_kwargs.update(kwargs)
        return call_kwargs
    
    @staticmethod
    def prepare_call_kwargs_i2v(pipeline, batch, **kwargs):
        call_kwargs = {}
        if "batch_cond_imgs" not in kwargs:
            call_kwargs["batch_cond_imgs"] = batch["batch_cond_imgs"]
        if "batch_bbox" not in kwargs:
            call_kwargs["batch_bbox"] = batch["batch_bbox"]
        if "batch_part" not in kwargs:
            call_kwargs["batch_part"] = batch["batch_part"]
        if "batch_slat_coords" not in kwargs:
            call_kwargs["batch_slat_coords"] = batch["batch_slat_coords"]
        if "batch_slat_feats" not in kwargs:
            call_kwargs["batch_slat_feats"] = batch["batch_slat_feats"]
        if "batch_id" not in kwargs:
            call_kwargs["batch_id"] = batch["batch_id"]
        if "generator" not in kwargs:
            call_kwargs["generator"] = torch.Generator(device=pipeline.device).manual_seed(pipeline.pipeline_config.seed)
        call_kwargs.update(pipeline.pipeline_config.call)
        call_kwargs.update(kwargs)
        return call_kwargs
    
    @staticmethod
    def prepare_call_kwargs_i2v_from_s1(pipeline, batch, **kwargs):
        batch_id = batch['batch_id'][0]
        s1_save_dir = pipeline.pipeline_config.s1_save_dir
        load_path = os.path.join(s1_save_dir, batch_id)
        call_kwargs = {}
        assert os.path.exists(os.path.join(load_path,'eval_box.npy')) and os.path.exists(os.path.join(load_path,'eval_decoded_voxel_0.npy'))
        load_box = np.load(os.path.join(load_path,'eval_box.npy')) # np, 2, 3
        load_part = np.load(os.path.join(load_path,'eval_decoded_voxel_0.npy')) 
        num_parts = len(load_box)
        slat_coords = []
        slat_feats = []
        boxes = []
        for i in range(num_parts):
            inds = load_part[:, 0] == i
            coords = load_part[inds, 1:] # n,3
            coords = torch.from_numpy(coords)
            if len(coords) == 0:
                print('---find empty parts---')
                continue
            else:
                boxes.append(torch.from_numpy(load_box[i]))
            ### inverse rotate to align box
            pts = (coords + 0.5) / 64 - 0.5 # -0.5, + 0.5
            # rotate
            matrix = np.eye(3)
            matrix[1,1] = 0
            matrix[1,2] = -1
            matrix[2,2] = 0
            matrix[2,1] = 1
            matrix = torch.from_numpy(matrix).float()
            pts = torch.matmul(pts, matrix.T)[:, :3]
            # voxelize
            coords = voxelize_point_cloud(pts.numpy())
            ###
            feats = torch.zeros(coords.shape[0], 8) # n, 8
            slat_coords.append(torch.from_numpy(coords).int().cuda())
            slat_feats.append(feats.cuda())
        
        boxes = torch.stack(boxes).numpy()
        load_box = boxes

        call_kwargs["batch_slat_coords"] = [slat_coords]
        call_kwargs["batch_slat_feats"] = [slat_feats]
        call_kwargs["batch_bbox"] = [torch.from_numpy(load_box).float().cuda()]
        call_kwargs["batch_id"] = [batch_id]
        call_kwargs["batch_cond_imgs"] = batch["batch_cond_imgs"]

        call_kwargs["save_gt"] = False

        if "generator" not in kwargs:
            call_kwargs["generator"] = torch.Generator(device=pipeline.device).manual_seed(pipeline.pipeline_config.seed)
        call_kwargs.update(pipeline.pipeline_config.call)
        call_kwargs.update(kwargs)
        return call_kwargs

    def val_t2v(self, batch):
        call_kwargs = self.prepare_call_kwargs(self, batch)
        # return self(**call_kwargs).images.mul(2).sub(1)
        return self(**call_kwargs)

    def val_i2v(self, batch):
        call_kwargs = self.prepare_call_kwargs_i2v(self, batch)
        # return self(**call_kwargs).images.mul(2).sub(1)
        return self(**call_kwargs)

    def val_i2v2(self, batch):
        call_kwargs = self.prepare_call_kwargs_i2v_from_s1(self, batch)
        # return self(**call_kwargs).images.mul(2).sub(1)
        return self(**call_kwargs)

    def val_t2v_16_9(self, batch):
        return self.t2v_by_ratio(batch, 16 / 9)

    def val_t2v_9_16(self, batch):
        return self.t2v_by_ratio(batch, 9 / 16)

    def t2v_by_ratio(self, batch, ratio):
        call_kwargs = self.prepare_call_kwargs(self, batch)
        spatial_unit_size = (
            self.pipeline_config.transformer_config.vae_scale_factor
            * self.pipeline_config.transformer_config.patch_size
            * (
                self.pipeline_config.transformer_config.temporal_attention_config.stfit_patch_size[-1]
                if hasattr(self.pipeline_config.transformer_config, "temporal_attention_config")
                and self.pipeline_config.transformer_config.temporal_attention_config is not None
                else 1
            )
        )

        area = call_kwargs["height"] * call_kwargs["width"]
        width = math.sqrt(area / ratio) * ratio
        height = round(width / ratio) // spatial_unit_size * spatial_unit_size
        width = round(width) // spatial_unit_size * spatial_unit_size
        call_kwargs.update(height=height, width=width)
        return self(**call_kwargs).images.mul(2).sub(1)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: str = "",
        num_inference_steps: int = 20,
        timesteps: List[int] = None,
        guidance_scale: float = 4.5,
        height: Optional[int] = None,
        width: Optional[int] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        prompt_attention_mask: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_attention_mask: Optional[torch.FloatTensor] = None,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        clean_caption: bool = True,
        timestep_shift: Optional[float] = None,
        batch_cond_imgs: Optional[torch.FloatTensor] = None,
        batch_bbox: Optional[torch.FloatTensor] = None,
        batch_part: Optional[torch.FloatTensor] = None,
        batch_slat_feats: Optional[torch.FloatTensor] = None,
        batch_slat_coords: Optional[torch.FloatTensor] = None,
        save_gt: Optional[bool] = False,
        batch_id=None,
        save_dir="./sample_results_s2",
        decode_part_id=None,
    ) -> Union[ImagePipelineOutput, Tuple]:
        
        save_name = batch_id[0]
        save_path = os.path.join(save_dir, save_name)
        os.makedirs(save_path, exist_ok=True)
        # if os.path.exists(os.path.join(save_path, f"eval_batches/sample_pred_{len(batch_bbox[0]) - 1}.glb")):
        #     return None
        print("sampling results in: ", save_path)
        
        device = self._execution_device

        do_classifier_free_guidance = guidance_scale > 1.0
        timestep_shift = timestep_shift or self.pipeline_config.timestep_shift

        # height = height // self.vae_scale_factor * self.vae_scale_factor
        # width = width // self.vae_scale_factor * self.vae_scale_factor
        # self.text_encoder.to(torch.bfloat16)
        # (prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask, prompt_embeds_pooled, negative_prompt_embeds_pooled) = (
        #     self.encode_prompt(
        #         prompt,
        #         do_classifier_free_guidance,
        #         negative_prompt=negative_prompt,
        #         # num_images_per_prompt=num_images_per_prompt,
        #         device=device,
        #         prompt_embeds=prompt_embeds,
        #         negative_prompt_embeds=negative_prompt_embeds,
        #         prompt_attention_mask=prompt_attention_mask,
        #         negative_prompt_attention_mask=negative_prompt_attention_mask,
        #         clean_caption=clean_caption,
        #         max_sequence_length=self.pipeline_config.max_sequence_length,
        #     )
        # )

        # if do_classifier_free_guidance:
        #     prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        #     prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)
        #     if self.pipeline_config.clip_ckpt_path is not None:
        #         prompt_embeds_pooled = torch.cat([negative_prompt_embeds_pooled, prompt_embeds_pooled], dim=0)

        img_conds = batch_cond_imgs
        img_conds = torch.cat(img_conds, dim=0)
        save_image(denormalize(img_conds[:, :3]), os.path.join(save_path,'eval_cond_img.png'))
        assert isinstance(batch_bbox, List), "batch box should be list"
        num_parts = len(batch_bbox[0])
        print("====np====", num_parts)
        img_conds = self.voxel_vae.encode_cond_img_from_tensor(img_conds)
        neg_img_conds = torch.zeros_like(img_conds)
        if do_classifier_free_guidance:
            img_conds = torch.cat([neg_img_conds, img_conds], dim=0)
            batch_bbox = [batch_bbox[0], batch_bbox[0]]
        assert img_conds.shape[0] == 2
        img_conds = repeat(img_conds, 'b n c -> (b np) n c', np=num_parts)
        # prompt_embeds = prompt_embeds.to(self.transformer.dtype)
        # prompt_attention_mask = prompt_attention_mask.to(self.transformer.dtype)
        # if self.pipeline_config.clip_ckpt_path is not None:
        #     prompt_embeds_pooled = prompt_embeds_pooled.to(self.transformer.dtype)

        # 5. Prepare latents.
        # latent_channels = self.transformer.config.in_channels
        # latents = self.prepare_latents(
        #     1,
        #     latent_channels,
        #     num_frames + (num_frames - 1) // self.vae.config.segment_size * (self.pipeline_config.transformer_config.vae_temporal_scale_factor - 1),
        #     height,
        #     width,
        #     prompt_embeds.dtype,
        #     device,
        #     generator,
        #     latents,
        # )
        assert do_classifier_free_guidance
        slat_coords_cat = []
        slat_feats_cat = []
        gt_slat_feats_cat = []
        for i in range(num_parts):
            ## decode slat
            slat_feats = torch.randn_like(batch_slat_feats[0][i])
            slat_coords = batch_slat_coords[0][i]
            batch_ids = torch.ones(slat_coords.shape[0], 1, dtype=slat_coords.dtype, device=slat_coords.device) * i
            slat_coords = torch.cat([batch_ids, slat_coords], dim=-1)
            slat_coords_cat.append(slat_coords)
            slat_feats_cat.append(slat_feats)
            gt_slat_feats_cat.append(batch_slat_feats[0][i])
        
        slat_coords_cat = torch.cat(slat_coords_cat)
        slat_feats_cat = torch.cat(slat_feats_cat)
        gt_slat_feats_cat = torch.cat(gt_slat_feats_cat)

        slat = sp.SparseTensor(
                    feats = slat_feats_cat.float(),
                    coords = slat_coords_cat,
                ).cuda()
        slat = slat.to(self.transformer.voxel_branch.input_layer.weight.dtype)
        slat = slat.to(self.transformer.voxel_branch.input_layer.weight.device)

        if save_gt:
            gt_slat = sp.SparseTensor(
                        feats = gt_slat_feats_cat.float(),
                        coords = slat_coords_cat,
                    ).cuda()

        if do_classifier_free_guidance:
            slat_coords_cat_cfg = torch.cat([slat_coords_cat[:, 0:1] + num_parts, slat_coords_cat[:, 1:]], dim=-1)
            slat_coords_cat_cfg = torch.cat([slat_coords_cat, slat_coords_cat_cfg])

        # voxel_latents = torch.randn(1, 8, num_parts, 16, 16, 16).to(device).to(img_conds.dtype) # b, c, f, h, w, d
        voxel_latents = slat

        # 6.1 Prepare micro-conditions.
        # added_cond_kwargs = self.prepare_added_cond_kwargs(
        #     1,
        #     fps,
        #     num_frames,
        #     height, 
        #     width, 
        #     prompt_embeds.dtype,
        #     device,
        #     1,
        #     do_classifier_free_guidance,
        # )
        # added_cond_kwargs["prompt_embeds_pooled"] = prompt_embeds_pooled

        # 7. Denoising loop
        timesteps_all = torch.linspace(1.0, 0, num_inference_steps + 1, device=voxel_latents.device)
        timesteps_all = timestep_shift * timesteps_all / (1 - timesteps_all + timestep_shift * timesteps_all)
        dts = timesteps_all[:-1] - timesteps_all[1:]
        timesteps = timesteps_all[:-1]

        # process conditions
        # condition_order_list = self.pipeline_config.condition_order.split(",")
        
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input_voxel = sp.SparseTensor(
                                                feats = torch.cat([voxel_latents.feats]*2),
                                                coords = slat_coords_cat_cfg,
                                            ).to(voxel_latents.device).to(voxel_latents.dtype)
                
                if self.pipeline_config.match_snr: # False
                    scale_factor = latents.shape[2] ** 0.5
                    current_timestep = t / (scale_factor - scale_factor * t + t)
                else:
                    current_timestep = t

                if not torch.is_tensor(current_timestep):
                    # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                    # This would be a good case for the `match` statement (Python 3.10+)
                    is_mps = latent_model_input_voxel.device.type == "mps"
                    if isinstance(current_timestep, float):
                        dtype = torch.float32 if is_mps else torch.float64
                    else:
                        dtype = torch.int32 if is_mps else torch.int64
                    current_timestep = torch.tensor([current_timestep], dtype=dtype, device=latent_model_input_voxel.device)
                elif len(current_timestep.shape) == 0:
                    current_timestep = current_timestep[None].to(latent_model_input_voxel.device)
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                # current_timestep = current_timestep.expand(latent_model_input_voxel.shape[0])
                current_timestep = current_timestep.expand(2) # cfg

                current_timestep = repeat(current_timestep, "b -> (b np)", np=num_parts)
                # predict noise model_output

                v_pred, v_pred_2 = self.transformer(
                    None,
                    # encoder_hidden_states=prompt_embeds,
                    # encoder_attention_mask=prompt_attention_mask,
                    timestep=current_timestep * 999,
                    # added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                    noisy_voxel_latent=latent_model_input_voxel,
                    img_conds = img_conds,
                    bboxes = batch_bbox,
                    np = num_parts,
                    batch_size=2,
                )

                # perform guidance
                if do_classifier_free_guidance:
                    # v_pred_uncond, v_pred_text = v_pred.chunk(2)
                    # v_pred = v_pred_uncond + guidance_scale * (v_pred_text - v_pred_uncond)
                    # voxel
                    v_pred_uncond_2_feats = v_pred_2.feats[:v_pred_2.feats.shape[0]//2]
                    v_pred_img_2_feats = v_pred_2.feats[v_pred_2.feats.shape[0]//2:]
                    v_pred_2 = slat.replace(v_pred_uncond_2_feats + guidance_scale * (v_pred_img_2_feats - v_pred_uncond_2_feats))

                # compute previous image: x_t -> x_t-1
                # latents = latents - dts[i] * v_pred
                voxel_latents = voxel_latents - dts[i] * v_pred_2

                # call the callback, if provided
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    step_idx = i
                    callback(step_idx, t, latents)

        if self.transformer.slat_norm:
            slat_normalization = {
                "mean": [
                    -2.1687545776367188,
                    -0.004347046371549368,
                    -0.13352349400520325,
                    -0.08418072760105133,
                    -0.5271206498146057,
                    0.7238689064979553,
                    -1.1414450407028198,
                    1.2039363384246826
                ],
                "std": [
                    2.377650737762451,
                    2.386378288269043,
                    2.124418020248413,
                    2.1748552322387695,
                    2.663944721221924,
                    2.371192216873169,
                    2.6217446327209473,
                    2.684523105621338
                ]
            }
            std = torch.tensor(slat_normalization['std'])[None].to(voxel_latents.device).to(voxel_latents.feats.dtype)
            mean = torch.tensor(slat_normalization['mean'])[None].to(voxel_latents.device).to(voxel_latents.feats.dtype)
            voxel_latents = voxel_latents * std + mean
            if save_gt:
                gt_slat = gt_slat * std + mean

        with torch.no_grad():
                # outputs = self.voxel_vae.decode(voxel_latents.to(torch.float32), formats=['mesh', 'gaussian'])
                outputs = []
                gt_outputs = []
                assert num_parts == voxel_latents.shape[0]
                for i in range(num_parts):
                    outs = self.voxel_vae.decode(voxel_latents.to(torch.float32)[i], formats=['mesh', 'gaussian'])
                    outputs.append(outs)
                    if save_gt:
                        gt_outs = self.voxel_vae.decode(gt_slat.to(torch.float32)[i], formats=['mesh', 'gaussian'])
                        gt_outputs.append(gt_outs)

        os.makedirs(os.path.join(save_path, 'eval_batches'), exist_ok=True)
        
        decode_part_id = range(len(outputs)) if decode_part_id is None else decode_part_id
        for i in decode_part_id:
            with torch.enable_grad():
                glb = postprocessing_utils.to_glb(
                    outputs[i]['gaussian'][0],
                    outputs[i]['mesh'][0],
                    # Optional parameters
                    simplify=0.95,          # Ratio of triangles to remove in the simplification process
                    texture_size=1024,      # Size of the texture used for the GLB
                    box=batch_bbox[0][i].float().cpu().numpy(), # 2,3
                )
            glb.export(os.path.join(save_path, f"eval_batches/sample_pred_{i}.glb"))
        
        if save_gt:
            for i in range(len(gt_outputs)):
                with torch.enable_grad():
                    glb = postprocessing_utils.to_glb(
                        gt_outputs[i]['gaussian'][0],
                        gt_outputs[i]['mesh'][0],
                        # Optional parameters
                        simplify=0.95,          # Ratio of triangles to remove in the simplification process
                        texture_size=1024,      # Size of the texture used for the GLB
                        box=batch_bbox[0][i].float().cpu().numpy(), # 2,3
                    )
                glb.export(os.path.join(save_path, f"eval_batches/sample_gt_{i}.glb"))
        
        print("===saved===")
        return None

        # Offload all models
        # self.maybe_free_model_hooks()
