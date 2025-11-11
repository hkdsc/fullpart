import math
from dataclasses import dataclass, field
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import torch
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
from einops import rearrange, repeat
import numpy as np

from ..configs.config_utils import to_immutable_dict
from ..models import VisualTokenizerConfig, VoxelTokenizerConfig, VoxelTokenizer
from ..models.transformers import Transformer2DModelConfig
from ..utils import measure_time
from .base_pipeline import EDMTrainConfig, PipelineMixin
from .t2v_pipeline_pixart_alpha import T2VPixArtAlphaPipeline, T2VPixArtAlphaPipelineConfig
from ..utils import pack_data, pack_data_voxel
import copy
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer
from diffusers.models import AutoencoderKL, Transformer2DModel
from diffusers.schedulers import DPMSolverMultistepScheduler
from torchvision.utils import save_image

def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    denormalized = tensor.clone()
    for c in range(3):
        denormalized[:, c] = denormalized[:, c] * std[c] + mean[c]
    return denormalized


@dataclass
class JointDiTSingle3DPipelineConfig(T2VPixArtAlphaPipelineConfig):
    """Configuration for Pipeline instantiation"""

    _target: Type = field(default_factory=lambda: JointDiTSingle3DPipeline)
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
            "num_inference_steps": 50,
        }
    )
    """The inference call arguments for the pipeline."""
    measure_time: bool = False

    

class JointDiTSingle3DPipeline(T2VPixArtAlphaPipeline, PipelineMixin):

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
        print("===convert vae to bf16===")
        self.voxel_vae.convert_to_bf16()
        print("===convert transformer to bf16===")
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
                    # print("===frame_voxel===", frame_voxel.shape)
                    latent = self.voxel_vae.encode(frame_voxel) # 8, 16, 16, 16
                    latent_list.append(latent)
                latent = torch.stack(latent_list, dim=1)[None] # 1, 8, 77, 16, 16, 16
            else:
                # sample is a static voxel
                latent = self.voxel_vae.encode(sample)
                latent = latent.unsqueeze(1).unsqueeze(0)
            latents.append(latent) 
            # print("===finall voxel latent in one batch===", latent.shape)
        latents, masks = pack_data_voxel(latents)
        return latents, masks


    def forward(self, batch):
        batch_part = batch['batch_part']
        batch_bbox = batch['batch_bbox']

        # print("=====batch part===", len(batch_part), len(batch_part[0]), batch_part[0][0].dtype)
        # print("=====batch bbox===", len(batch_bbox), len(batch_bbox[0]))

        # get img conds
        if "batch_cond_imgs" in batch.keys():
            img_conds = batch["batch_cond_imgs"]
            img_conds = torch.cat(img_conds, dim=0)
            # print("=========img conds==========", img_conds.shape)
            img_conds = self.voxel_vae.encode_cond_img_from_tensor(img_conds)
        else:
            img_conds = None
        
        # drop img conds for cfg
        assert self.transformer.drop_img_conds > 0.
        import random
        if random.random() < self.transformer.drop_img_conds:
            img_conds = torch.zeros_like(img_conds)

        # encode part voxel
        voxel_latents, voxel_masks = self.get_latents_voxel(batch_part) # 1, 8, np, 16, 16, 16
        num_parts = voxel_latents.shape[2]
        batch_size = voxel_latents.shape[0]
        voxel_latents = voxel_latents.to(self.voxel_vae.encoder.dtype)

        if "batch_voxels" in batch.keys():
            batch_voxels = batch["batch_voxels"]
            voxel_latents, voxel_masks = self.get_latents_voxel(batch_voxels) # b, 8, 77, 16, 16, 16
            voxel_latents = voxel_latents.to(self.voxel_vae.encoder.dtype)
        
        eps_2 = self.pipeline_config.epsilon_voxel
        z_2 = self.generate_noise(voxel_latents)

        if "t_step" in batch:
            t = batch["t_step"].repeat_interleave(batch_size).to(device=voxel_latents.device, dtype=voxel_latents.dtype)
        else:
            if self.pipeline_config.logit_normal:
                t_logit = torch.exp(torch.randn(batch_size, device=z_2.device))
                t = t_logit / (t_logit + 1)
            else:
                t = torch.rand(batch_size, device=z_2.device)

            t = self.pipeline_config.timestep_shift * t / (1 - t + self.pipeline_config.timestep_shift * t)

            timestep = t
            if self.pipeline_config.match_snr:
                scale_factor = voxel_latents.shape[2] ** 0.5
                t = scale_factor * t / (1 - t + scale_factor * t)
        
        timestep = repeat(timestep, "b -> (b np)", np=num_parts)
        t_expand_2 = t[:, None, None, None, None, None]
        z_t_2 = (1 - t_expand_2) * voxel_latents + (eps_2 + (1 - eps_2) * t_expand_2) * z_2
        u_2 = (1 - eps_2) * z_2 - voxel_latents

        z_t_2 = z_t_2.to(voxel_latents.dtype)
        u_2 = u_2.to(voxel_latents.dtype)

        with measure_time("Transformer", self.pipeline_config.measure_time):
            v, v_2 = self.transformer(
                None,
                timestep=timestep * 999,
                return_dict=False,
                noisy_voxel_latent = z_t_2,
                img_conds = img_conds,
                bboxes = batch_bbox,
                np=num_parts,
            )
        
        self.cnt += 1
        return self.compute_loss({"pred_voxel": v_2, "target_voxel": u_2})
    
    def compute_loss(self, result_dict):
        # mask shape = b 1 f h w
        # pred, target = result_dict["pred"], result_dict["target"]
        pred_voxel, target_voxel = result_dict["pred_voxel"], result_dict["target_voxel"]
        loss_weight_voxel = 1.
        loss_voxel = ((pred_voxel.float() - target_voxel.float()) ** 2).reshape(target_voxel.shape[0], -1).mean(dim=1)
        loss_reweight_voxel = (loss_voxel * loss_weight_voxel).mean()  # MSE loss
        loss_orginal_voxel = loss_voxel.mean()  # MSE loss

        total_loss =  loss_reweight_voxel
        # print('===loss total, loss 3d==', total_loss.item(), loss_orginal_voxel.item())
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
        # if "prompt" not in kwargs:
        #     # call_kwargs["prompt"] = batch["prompts"]
        #     call_kwargs["prompt"] = batch["batch_caption"]
        if "batch_cond_imgs" not in kwargs:
            call_kwargs["batch_cond_imgs"] = batch["batch_cond_imgs"]
        if "batch_bbox" not in kwargs:
            call_kwargs["batch_bbox"] = batch["batch_bbox"]
        if "batch_part" not in kwargs:
            call_kwargs["batch_part"] = batch["batch_part"]
        if "batch_id" not in kwargs:
            call_kwargs["batch_id"] = batch["batch_id"]
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
        batch_id=None,
        save_dir="./sample_results",
    ) -> Union[ImagePipelineOutput, Tuple]:
        
        
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

        print('----save_name---', batch_id)
        save_name = batch_id[0]
        save_path = os.path.join(save_dir, save_name)
        os.makedirs(save_path, exist_ok=True)

        img_conds = batch_cond_imgs
        print("===save cond img===")
        img_conds = torch.cat(img_conds, dim=0)
        save_image(denormalize(img_conds[:, :3]), os.path.join(save_path, 'eval_cond_img.png'))
        assert isinstance(batch_bbox, List), "batch box should be list"
        num_parts = len(batch_bbox[0])
        print("====np====", num_parts)
        img_conds = self.voxel_vae.encode_cond_img_from_tensor(img_conds)
        neg_img_conds = torch.zeros_like(img_conds)
        if do_classifier_free_guidance:
            img_conds = torch.cat([neg_img_conds, img_conds], dim=0)
            batch_bbox = [batch_bbox[0], batch_bbox[0]]
        # prompt_embeds = prompt_embeds.to(self.transformer.dtype)
        # prompt_attention_mask = prompt_attention_mask.to(self.transformer.dtype)
        # if self.pipeline_config.clip_ckpt_path is not None:
        #     prompt_embeds_pooled = prompt_embeds_pooled.to(self.transformer.dtype)

        # 5. Prepare latents.
        # latent_channels = self.transformer.config.in_channels
        voxel_latents = torch.randn(1, 8, num_parts, 16, 16, 16).to(device).to(img_conds.dtype) # b, c, f, h, w, d

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
        
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input_voxel = torch.cat([voxel_latents] * 2) if do_classifier_free_guidance else latents                
                
                if self.pipeline_config.match_snr: # False
                    scale_factor = latents.shape[2] ** 0.5
                    current_timestep = t / (scale_factor - scale_factor * t + t)
                else:
                    current_timestep = t

                if not torch.is_tensor(current_timestep):
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
                current_timestep = current_timestep.expand(latent_model_input_voxel.shape[0])

                current_timestep = repeat(current_timestep, "b -> (b np)", np=num_parts)

                v_pred, v_pred_2 = self.transformer(
                    None,
                    timestep=current_timestep * 999,
                    return_dict=False,
                    noisy_voxel_latent=latent_model_input_voxel,
                    img_conds = img_conds,
                    bboxes = batch_bbox,
                    np = num_parts,
                )

                # perform guidance
                if do_classifier_free_guidance:
                    v_pred_uncond_2, v_pred_img_2 = v_pred_2.chunk(2)
                    v_pred_2 = v_pred_uncond_2 + guidance_scale * (v_pred_img_2 - v_pred_uncond_2)

                voxel_latents = voxel_latents - dts[i] * v_pred_2

                # call the callback, if provided
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    step_idx = i
                    callback(step_idx, t, latents)

        # decode voxel
        print("========eval decoded voxel=====", voxel_latents.shape) # 1,8,1,16,16,16
        save_decode_voxel = []
        for i, voxel_latent in enumerate(voxel_latents):
            decoded_voxel = self.voxel_vae.decode(voxel_latent.permute(1, 0, 2, 3, 4).float()) # -> 77, 8, 16, 16, 16
            save_decode_voxel.append(decoded_voxel)
            np.save(os.path.join(save_path, f'eval_decoded_voxel_{i}.npy'), decoded_voxel.detach().cpu().numpy())
            # np.save(f'debug/eval_box.npy', batch_bbox[0].detach().cpu().numpy())
            print(f"===saved to eval decoded voxel {i}===", decoded_voxel.shape) 
        del decoded_voxel
        del save_decode_voxel
        del voxel_latents

        save_dict = {}
        save_dict_grid = {}
        all_res = [16, 16, 16]
        coords = torch.meshgrid(*[torch.arange(res) for res in all_res], indexing='ij')
        coords = torch.stack(coords, dim=-1).reshape(-1, 3)
        # get global coords
        coords_pts = (coords + 0.5) / all_res[0] - 0.5 # [-0.5, 0.5]
        coords_pts = coords_pts.to(img_conds.device)
        for bs_id, part_verts in enumerate(batch_part):
            # bs = 1 for now
            for i in range(num_parts):
                box = batch_bbox[bs_id][i] # 2, 3
                scale = (box[1] - box[0]).max()
                center = box.mean(dim=0)
                part_verts_global = part_verts[i] * scale + center
                coords_global = coords_pts * scale + center
                save_dict[f'{i}'] = part_verts_global.to(torch.float32).detach().cpu().numpy()
                save_dict_grid[f'{i}'] = coords_global.to(torch.float32).detach().cpu().numpy()

        np.savez(os.path.join(save_path,'eval_part_decoded_global.npz'), **save_dict)
        np.savez(os.path.join(save_path,'eval_part_decoded_global_grid.npz'), **save_dict_grid)
        np.save(os.path.join(save_path, 'eval_box.npy'), batch_bbox[0].to(torch.float32).detach().cpu().numpy())
        print("===save box to===", save_name)

        return None

        # Offload all models
        # self.maybe_free_model_hooks()
