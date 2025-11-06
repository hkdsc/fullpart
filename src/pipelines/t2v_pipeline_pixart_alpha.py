import inspect
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import open_clip
import torch
from diffusers.models import AutoencoderKL, Transformer2DModel
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import ASPECT_RATIO_512_BIN, ASPECT_RATIO_1024_BIN, PixArtAlphaPipeline, logger, retrieve_timesteps
from diffusers.schedulers import DPMSolverMultistepScheduler
from diffusers.utils import deprecate
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5Tokenizer

from ..configs.config_utils import to_immutable_dict
from ..models import VisualTokenizer
from ..models.text_encoder.utils import encode_prompt_clip_g, encode_prompt_clip_l
from ..models.transformers import Transformer2DModelConfig
from ..utils import measure_time, recover_model, partition_model, TrainingStatus
from .base_pipeline import BasePipelineConfig, EDMTrainConfig, PipelineMixin


@dataclass
class T2VPixArtAlphaPipelineConfig(BasePipelineConfig):
    """Configuration for Pipeline instantiation"""

    _target: Type = field(default_factory=lambda: T2VPixArtAlphaPipeline)
    """target class to instantiate"""

    ckpt_path: str = None

    edm_config: EDMTrainConfig = EDMTrainConfig(
        P_mean=-0.4,
        P_std=1.0,
        c_in_type="EDM",
        c_noise_type="EDM",
        c_skip_type="EDM",
        c_out_type="VPred",
        loss_weight_type="EDM",
        noise_dist_type="EDM",
    )
    """The EDM config for the pipeline."""

    transformer_config: Optional[Transformer2DModelConfig] = None
    """The transformer config for the pipeline."""

    proportion_empty_prompts: float = 0.0
    """Proportion of empty prompts to use."""

    max_sequence_length: int = 120
    """The maximum number of tokens for text encoding"""

    call: Dict[str, Any] = to_immutable_dict(
        {
            "num_frames": 16,
            "height": 256,
            "width": 384,
            "num_inference_steps": 50,
        }
    )
    """The inference call arguments for the pipeline."""


class T2VPixArtAlphaPipeline(PixArtAlphaPipeline, PipelineMixin):
    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        vae: AutoencoderKL,
        transformer: Transformer2DModel,
        scheduler: DPMSolverMultistepScheduler,
        clip_g_model: open_clip.CLIP = None,
        clip_l_model: CLIPTextModel = None,
        tokenizer_clip: CLIPTokenizer = None,
    ):
        super().__init__(tokenizer, text_encoder, vae, transformer, scheduler)

        self.register_modules(clip_g_model=clip_g_model, clip_l_model=clip_l_model, tokenizer_clip=tokenizer_clip)

    def forward(self, batch):
        with measure_time("VAE", self.pipeline_config.measure_time):
            latents = self.get_latents(batch)[0]
        batch_size = latents.shape[0]

        added_cond_kwargs = self.prepare_added_cond_kwargs(
            batch_size,
            fps=batch["sample_fps"],
            num_frames=batch["num_frames"],
            height=[height for height, _ in batch["target_sizes"]],
            width=[width for _, width in batch["target_sizes"]],
            dtype=latents.dtype,
            device=latents.device,
        )

        with measure_time("Text", self.pipeline_config.measure_time):
            dtype = self.transformer.pos_embed.proj.weight.dtype
            prompt_embeds, prompt_attention_mask, prompt_masks = self.get_t5_prompt_embeddings(
                prompts=batch.get("prompts", None),
                t5_prompt_embeds_list=batch.get("t5_prompt_embeds", None),
                device=latents.device,
                dtype=dtype,
            )

            prompt_embeds_pooled = None
            if self.pipeline_config.clip_ckpt_path is not None:
                prompt_embeds_clip, prompt_attention_mask_clip, prompt_embeds_pooled = self.get_clip_prompt_embeddings(
                    prompts=batch.get("prompts", None),
                    clip_prompt_embeds_list=batch.get("clip_prompt_embeds", None),
                    device=latents.device,
                    dtype=dtype,
                    prompt_masks=prompt_masks,
                    target_size=prompt_embeds.shape[2],
                )

                prompt_embeds = torch.cat((prompt_embeds, prompt_embeds_clip.to(prompt_embeds.dtype)), dim=1)  # bs * 120+77 * 4096
                prompt_attention_mask = torch.cat([prompt_attention_mask, prompt_attention_mask_clip.to(prompt_attention_mask.dtype)], dim=1)

            prompt_embeds = prompt_embeds.to(dtype)
            prompt_attention_mask = prompt_attention_mask.to(dtype)
            if prompt_embeds_pooled is not None:
                prompt_embeds_pooled = prompt_embeds_pooled.to(dtype)

        added_cond_kwargs["prompt_embeds_pooled"] = prompt_embeds_pooled

        with measure_time("Transformer", self.pipeline_config.measure_time):
            scaled_input, scaled_sigmas, target, loss_weight = self.edm_step(batch_size, latents)
            model_output = self.transformer(
                scaled_input,
                timestep=scaled_sigmas,
                encoder_hidden_states=prompt_embeds,
                encoder_attention_mask=prompt_attention_mask,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]
        if self.transformer.config.out_channels // 2 == target.shape[1]:
            model_output = model_output.chunk(2, dim=1)[0]

        return self.compute_loss({"pred": model_output, "target": target, "loss_weight": loss_weight})

    def prepare_added_cond_kwargs(
        self, batch_size, fps, num_frames, height, width, dtype, device, num_images_per_prompt: int = 1, do_classifier_free_guidance: bool = False
    ):
        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
        if isinstance(fps, list):
            fps = torch.tensor(fps).repeat_interleave(num_images_per_prompt, 0).view(-1, 1)
        else:
            fps = torch.tensor(fps).repeat(batch_size * num_images_per_prompt, 1)

        if isinstance(num_frames, list):
            num_frames = torch.tensor(num_frames).repeat_interleave(num_images_per_prompt, 0).view(-1, 1)
        else:
            num_frames = torch.tensor(num_frames).repeat(batch_size * num_images_per_prompt, 1)

        if isinstance(height, list) and isinstance(width, list):
            resolution = torch.tensor([height, width]).transpose(0, 1).repeat_interleave(num_images_per_prompt, 0)
            aspect_ratio = torch.tensor([h / w for h, w in zip(height, width)]).repeat_interleave(num_images_per_prompt, 0).view(-1, 1)
        elif isinstance(height, int) and isinstance(width, int):
            resolution = torch.tensor([height, width]).repeat(batch_size * num_images_per_prompt, 1)
            aspect_ratio = torch.tensor([height / width]).repeat(batch_size * num_images_per_prompt, 1)
        else:
            raise NotImplementedError

        if do_classifier_free_guidance:
            fps = torch.cat([fps, fps], dim=0)
            num_frames = torch.cat([num_frames, num_frames], dim=0)
            resolution = torch.cat([resolution, resolution], dim=0)
            aspect_ratio = torch.cat([aspect_ratio, aspect_ratio], dim=0)

        added_cond_kwargs = {"fps": fps, "frames": num_frames, "resolution": resolution, "aspect_ratio": aspect_ratio}
        added_cond_kwargs = {k: v.to(dtype=dtype, device=device) for k, v in added_cond_kwargs.items()}
        return added_cond_kwargs

    @torch.no_grad()
    def get_t5_prompt_embeddings(self, prompts, t5_prompt_embeds_list, device, dtype):
        num_prompts = len(t5_prompt_embeds_list) if t5_prompt_embeds_list is not None else len(prompts)
        prompt_masks = [random.random() > self.pipeline_config.proportion_empty_prompts for i in range(num_prompts)]
        if prompts is not None:
            prompts = [prompt if mask else "" for mask, prompt in zip(prompt_masks, prompts)]
        if t5_prompt_embeds_list is None:
            self.text_encoder.to(torch.bfloat16)  # NOTE: nan when float16
            prompt_embeds, prompt_attention_mask, _, _ = super().encode_prompt(
                prompts,
                do_classifier_free_guidance=False,
                device=device,
                clean_caption=True,
                max_sequence_length=self.pipeline_config.max_sequence_length,
            )
        else:
            max_sequence_length = min(max(t.size(0) for t in t5_prompt_embeds_list), self.pipeline_config.max_sequence_length)
            prompt_embeds = torch.zeros((len(t5_prompt_embeds_list), max_sequence_length, t5_prompt_embeds_list[0].size(-1)), device=device, dtype=dtype)
            prompt_attention_mask = torch.zeros((len(t5_prompt_embeds_list), max_sequence_length), device=device, dtype=dtype)
            if hasattr(self, "empty_prompt_embed_t5"):
                empty_prompt_embed_t5 = self.empty_prompt_embed_t5
            else:
                empty_prompt_embed_t5 = torch.load("/video/zhengmingwu/ckpts/t5_empty.pt", map_location="cpu").to(device=device, dtype=dtype)
                self.empty_prompt_embed_t5 = empty_prompt_embed_t5
            for i, prompt_embed in enumerate(t5_prompt_embeds_list):
                if not prompt_masks[i]:
                    prompt_embeds[i, : empty_prompt_embed_t5.size(0)] = empty_prompt_embed_t5[: empty_prompt_embed_t5.size(0)]
                    prompt_attention_mask[i, : empty_prompt_embed_t5.size(0)] = 1
                else:
                    prompt_embeds[i, : prompt_embed.size(0)] = prompt_embed[:max_sequence_length]
                    prompt_attention_mask[i, : prompt_embed.size(0)] = 1

        return prompt_embeds, prompt_attention_mask, prompt_masks

    @torch.no_grad()
    def get_clip_prompt_embeddings(self, prompts, clip_prompt_embeds_list, device, dtype, prompt_masks, target_size):
        if clip_prompt_embeds_list is None:
            # 我们可以更优雅地使用 register_forward_pre_hook 或者 context manager 来做这件事，但是这会导致难以精细的 schedule 通信和其他计算
            if self.pipeline_config.distributed_clip and self.train_status == TrainingStatus.TRAINING:
                recover_model(self.clip_l_model)

            (
                prompt_embeds_clip_l,
                prompt_attention_mask_clip,
                negative_prompt_embeds_clip_l,
                negative_prompt_attention_mask_clip_l,
                prompt_embeds_pooled_clip_l,
                negative_prompt_embeds_pooled_clip_l,
            ) = encode_prompt_clip_l(self, prompts, do_classifier_free_guidance=False, device=device, clean_caption=True, max_length=77)

            if self.pipeline_config.distributed_clip and self.train_status == TrainingStatus.TRAINING:
                partition_model(self.clip_l_model)
                recover_model(self.clip_g_model)

            prompt_embeds_clip_g, negative_prompt_embeds_clip_g, prompt_embeds_pooled_clip_g, negative_prompt_embeds_pooled_clip_g = encode_prompt_clip_g(
                self, prompts, do_classifier_free_guidance=False, device=device, clean_caption=True, max_length=77
            )

            if self.pipeline_config.distributed_clip and self.train_status == TrainingStatus.TRAINING:
                partition_model(self.clip_g_model)

            prompt_embeds_clip = torch.cat((prompt_embeds_clip_l, prompt_embeds_clip_g), dim=2)  # bs * 120 * 2048
            prompt_embeds_pooled = torch.cat((prompt_embeds_pooled_clip_l, prompt_embeds_pooled_clip_g), dim=1)
        else:
            max_sequence_length = min(max(t.size(0) - 1 for t in clip_prompt_embeds_list), 77)
            prompt_embeds_clip = torch.zeros(
                (len(clip_prompt_embeds_list), max_sequence_length, clip_prompt_embeds_list[0].size(-1)), device=device, dtype=dtype
            )
            prompt_attention_mask_clip = torch.zeros((len(prompt_embeds_clip), max_sequence_length), device=device, dtype=dtype)
            prompt_embeds_pooled = torch.zeros((len(prompt_embeds_clip), clip_prompt_embeds_list[0].size(-1)), device=device, dtype=dtype)
            if hasattr(self, "empty_prompt_embed_clip"):
                empty_prompt_embed_clip = self.empty_prompt_embed_clip
            else:
                empty_prompt_embed_clip = torch.load("/video/yht/clip_empty.pt", map_location="cpu").to(device=device, dtype=dtype)
                self.empty_prompt_embed_clip = empty_prompt_embed_clip

            for i, prompt_embed in enumerate(clip_prompt_embeds_list):
                if not prompt_masks[i]:
                    prompt_embeds_clip[i, : empty_prompt_embed_clip.size(0) - 1] = empty_prompt_embed_clip[: empty_prompt_embed_clip.size(0) - 1]
                    prompt_attention_mask_clip[i, : empty_prompt_embed_clip.size(0) - 1] = 1
                    prompt_embeds_pooled[i] = empty_prompt_embed_clip[-1]
                else:
                    prompt_embeds_clip[i, : prompt_embed.size(0) - 1] = prompt_embed[: prompt_embed.size(0) - 1]
                    prompt_attention_mask_clip[i, : prompt_embed.size(0)] = 1
                    prompt_embeds_pooled[i] = prompt_embed[-1]

        prompt_embeds_clip = torch.cat(
            (
                prompt_embeds_clip,
                torch.zeros(
                    (prompt_embeds_clip.shape[0], prompt_embeds_clip.shape[1], target_size - prompt_embeds_clip.shape[2]),
                    device=prompt_embeds_clip.device,
                    dtype=prompt_embeds_clip.dtype,
                ),
            ),
            dim=2,
        )

        return prompt_embeds_clip, prompt_attention_mask_clip, prompt_embeds_pooled

    @staticmethod
    def prepare_call_kwargs(pipeline, batch, **kwargs):
        call_kwargs = {}
        if "prompt" not in kwargs:
            call_kwargs["prompt"] = batch["prompts"]
        if "generator" not in kwargs:
            call_kwargs["generator"] = torch.Generator(device=pipeline.device).manual_seed(pipeline.pipeline_config.seed)
        call_kwargs.update(pipeline.pipeline_config.call)
        call_kwargs.update(kwargs)
        return call_kwargs

    def val_t2v(self, batch):
        call_kwargs = self.prepare_call_kwargs(self, batch)
        return self(**call_kwargs).images.mul(2).sub(1)

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, latents=None):

        if hasattr(self.vae.config, "temporal_scale_factor"):
            latent_num_frames = (num_frames - 1) // self.vae.config.temporal_scale_factor + 1
        else:
            latent_num_frames = num_frames
        shape = (
            batch_size,
            num_channels_latents,
            latent_num_frames,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        do_classifier_free_guidance: bool = True,
        negative_prompt: str = "",
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        prompt_attention_mask: Optional[torch.FloatTensor] = None,
        negative_prompt_attention_mask: Optional[torch.FloatTensor] = None,
        clean_caption: bool = False,
        max_sequence_length: int = 120,  
        **kwargs,
    ):
        prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = super().encode_prompt(
            prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            device=device,
            clean_caption=clean_caption,
            max_sequence_length=max_sequence_length,
        )

        prompt_embeds_pooled = None
        negative_prompt_embeds_pooled = None
        if self.pipeline_config.clip_ckpt_path is not None:
            if self.pipeline_config.distributed_clip and self.train_status == TrainingStatus.TRAINING:
                recover_model(self.clip_l_model)

            (
                prompt_embeds_clip_l,
                prompt_attention_mask_clip_l,
                negative_prompt_embeds_clip_l,
                negative_prompt_attention_mask_clip_l,
                prompt_embeds_pooled_clip_l,
                negative_prompt_embeds_pooled_clip_l,
            ) = encode_prompt_clip_l(
                self,
                prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
                device=device,
                clean_caption=clean_caption,
                max_length=77,
            )

            if self.pipeline_config.distributed_clip and self.train_status == TrainingStatus.TRAINING:
                partition_model(self.clip_l_model)
                recover_model(self.clip_g_model)

            prompt_embeds_clip_g, negative_prompt_embeds_clip_g, prompt_embeds_pooled_clip_g, negative_prompt_embeds_pooled_clip_g = encode_prompt_clip_g(
                self,
                prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
                device=device,
                clean_caption=clean_caption,
                max_length=77,
            )

            if self.pipeline_config.distributed_clip and self.train_status == TrainingStatus.TRAINING:
                partition_model(self.clip_g_model)

            prompt_embeds_clip = torch.cat((prompt_embeds_clip_l, prompt_embeds_clip_g), dim=2)  # bs * 120 * 2048
            prompt_embeds_clip = torch.cat(
                (prompt_embeds_clip, torch.zeros((prompt_embeds_clip.shape[0], 77, 2048), device=prompt_embeds_clip.device, dtype=prompt_embeds_clip.dtype)),
                dim=2,
            )
            prompt_embeds_pooled = torch.cat((prompt_embeds_pooled_clip_l, prompt_embeds_pooled_clip_g), dim=1)

            prompt_embeds = torch.cat((prompt_embeds, prompt_embeds_clip), dim=1)  # bs * 120+77 * 4096
            prompt_attention_mask = torch.cat([prompt_attention_mask, prompt_attention_mask_clip_l], dim=1)

            if negative_prompt_embeds is not None:
                negative_prompt_embeds_clip = torch.cat((negative_prompt_embeds_clip_l, negative_prompt_embeds_clip_g), dim=2)
                negative_prompt_embeds_clip = torch.cat(
                    (
                        negative_prompt_embeds_clip,
                        torch.zeros((prompt_embeds_clip.shape[0], 77, 2048), device=prompt_embeds_clip.device, dtype=prompt_embeds_clip.dtype),
                    ),
                    dim=2,
                )
                negative_prompt_embeds_pooled = torch.cat((negative_prompt_embeds_pooled_clip_l, negative_prompt_embeds_pooled_clip_g), dim=1)
                negative_prompt_embeds = torch.cat((negative_prompt_embeds, negative_prompt_embeds_clip), dim=1)  # bs * 120+77 * 4096
                negative_prompt_attention_mask = torch.cat([negative_prompt_attention_mask, negative_prompt_attention_mask_clip_l], dim=1)

        return prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask, prompt_embeds_pooled, negative_prompt_embeds_pooled

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: str = "",
        num_inference_steps: int = 20,
        timesteps: List[int] = None,
        guidance_scale: float = 4.5,
        num_images_per_prompt: Optional[int] = 1,
        num_frames: int = 1,
        fps: float = 8.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        prompt_attention_mask: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_attention_mask: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pt",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        clean_caption: bool = True,
        **kwargs,
    ) -> Union[ImagePipelineOutput, Tuple]:
        if "mask_feature" in kwargs:
            deprecation_message = "The use of `mask_feature` is deprecated. It is no longer used in any computation and that doesn't affect the end results. It will be removed in a future version."
            deprecate("mask_feature", "1.0.0", deprecation_message, standard_warn=False)
        # 1. Check inputs. Raise error if not correct

        height = height or self.transformer.config.sample_size * self.vae_scale_factor
        width = width or self.transformer.config.sample_size * self.vae_scale_factor

        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            callback_steps,
            prompt_embeds,
            negative_prompt_embeds,
            prompt_attention_mask,
            negative_prompt_attention_mask,
        )

        # 2. Default height and width to transformer
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        prompt_embeds_pooled = None

        # 3. Encode input prompt
        self.text_encoder.to(torch.bfloat16)  # nan when float16
        (prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask, prompt_embeds_pooled, negative_prompt_embeds_pooled) = (
            self.encode_prompt(
                prompt,
                do_classifier_free_guidance,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                negative_prompt_attention_mask=negative_prompt_attention_mask,
                clean_caption=clean_caption,
                max_length=self.pipeline_config.max_sequence_length,  
            )
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)
            if self.pipeline_config.clip_ckpt_path is not None:
                prompt_embeds_pooled = torch.cat([negative_prompt_embeds_pooled, prompt_embeds_pooled], dim=0)

        prompt_embeds = prompt_embeds.to(self.transformer.pos_embed.proj.weight.dtype)
        prompt_attention_mask = prompt_attention_mask.to(self.transformer.pos_embed.proj.weight.dtype)
        if self.pipeline_config.clip_ckpt_path is not None:
            prompt_embeds_pooled = prompt_embeds_pooled.to(self.transformer.pos_embed.proj.weight.dtype)

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)

        # 5. Prepare latents.
        latent_channels = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            latent_channels,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.1 Prepare micro-conditions.
        added_cond_kwargs = self.prepare_added_cond_kwargs(
            batch_size,
            fps,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
        )
        added_cond_kwargs["prompt_embeds_pooled"] = prompt_embeds_pooled

        # 7. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                current_timestep = t
                if not torch.is_tensor(current_timestep):
                    # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                    # This would be a good case for the `match` statement (Python 3.10+)
                    is_mps = latent_model_input.device.type == "mps"
                    if isinstance(current_timestep, float):
                        dtype = torch.float32 if is_mps else torch.float64
                    else:
                        dtype = torch.int32 if is_mps else torch.int64
                    current_timestep = torch.tensor([current_timestep], dtype=dtype, device=latent_model_input.device)
                elif len(current_timestep.shape) == 0:
                    current_timestep = current_timestep[None].to(latent_model_input.device)
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                current_timestep = current_timestep.expand(latent_model_input.shape[0])
                # predict noise model_output
                noise_pred = self.transformer(
                    latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask,
                    timestep=current_timestep,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]
                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # iddpm sigma
                if self.transformer.config.out_channels // 2 == latent_channels:
                    noise_pred = noise_pred.chunk(2, dim=1)[0]
                else:
                    noise_pred = noise_pred

                # compute previous image: x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        if not output_type == "latent":
            if isinstance(self.vae, VisualTokenizer):
                # NOTE: our 3d vae always expects latents in the format b c f h w
                image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
                image = rearrange(image, "b c f h w -> (b f) c h w")
            else:
                latents = rearrange(latents, "b c f h w -> (b f) c h w")
                enable_vae_temporal_decoder = "num_frames" in set(inspect.signature(self.vae.forward).parameters.keys())
                if enable_vae_temporal_decoder:
                    image = self.decode_latents_with_temporal_decoder(latents / self.vae.config.scaling_factor)
                else:
                    image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        else:
            image = latents

        if not output_type == "latent":
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)

    def decode_latents_with_temporal_decoder(self, latents):
        video = []

        decode_chunk_size = 14
        for frame_idx in range(0, latents.shape[0], decode_chunk_size):
            num_frames_in = latents[frame_idx : frame_idx + decode_chunk_size].shape[0]

            decode_kwargs = {}
            decode_kwargs["num_frames"] = num_frames_in

            video.append(self.vae.decode(latents[frame_idx : frame_idx + decode_chunk_size], **decode_kwargs).sample)

        video = torch.cat(video)
        return video
