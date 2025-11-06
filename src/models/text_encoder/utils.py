from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import torch
from torch.utils.checkpoint import checkpoint
from diffusers.utils import deprecate
from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import logger
import open_clip
from open_clip.transformer import text_global_pool


def encode_with_transformer(model, text):
    x = model.token_embedding(text)  # [batch_size, n_ctx, d_model]
    x = x + model.positional_embedding
    x = x.permute(1, 0, 2)  # NLD -> LND
    x = text_transformer_forward(model, x, attn_mask=model.attn_mask)
    x = x.permute(1, 0, 2)  # LND -> NLD
    x = model.ln_final(x)
    x_pooled, _ = text_global_pool(x, text, model.text_pool_type)
    return x, x_pooled


def text_transformer_forward(model, x: torch.Tensor, attn_mask=None):
    for i, r in enumerate(model.transformer.resblocks):
        if i == len(model.transformer.resblocks):
            break
        if model.transformer.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint(r, x, attn_mask)
        else:
            x = r(x, attn_mask=attn_mask)
    return x


def encode_prompt_clip_l(
    pipeline,
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
    max_length: int = 77,  # NOTE(m2v)
    **kwargs,
):

    if "mask_feature" in kwargs:
        deprecation_message = "The use of `mask_feature` is deprecated. It is no longer used in any computation and that doesn't affect the end results. It will be removed in a future version."
        deprecate("mask_feature", "1.0.0", deprecation_message, standard_warn=False)

    if device is None:
        device = pipeline._execution_device

    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    if prompt_embeds is None:
        prompt = pipeline._text_preprocessing(prompt, clean_caption=clean_caption)
        text_inputs = pipeline.tokenizer_clip(
            prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids

        prompt_attention_mask = text_inputs.attention_mask
        prompt_attention_mask = prompt_attention_mask.to(device)

        prompt_embeds = pipeline.clip_l_model(text_input_ids.to(device), attention_mask=prompt_attention_mask)
        prompt_embeds_pooling = prompt_embeds[1]
        prompt_embeds = prompt_embeds[0]

    if pipeline.clip_l_model is not None:
        dtype = pipeline.clip_l_model.dtype
    elif pipeline.transformer is not None:
        dtype = pipeline.transformer.dtype
    else:
        dtype = None

    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    prompt_embeds_pooling = prompt_embeds_pooling.to(dtype=dtype, device=device)

    bs_embed, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
    prompt_attention_mask = prompt_attention_mask.view(bs_embed, -1)
    prompt_attention_mask = prompt_attention_mask.repeat(num_images_per_prompt, 1)
    prompt_embeds_pooling = torch.repeat_interleave(prompt_embeds_pooling, num_images_per_prompt, dim=0)

    # get unconditional embeddings for classifier free guidance
    if do_classifier_free_guidance and negative_prompt_embeds is None:
        uncond_tokens = [negative_prompt] * batch_size
        uncond_tokens = pipeline._text_preprocessing(uncond_tokens, clean_caption=clean_caption)
        max_length = prompt_embeds.shape[1]
        uncond_input = pipeline.tokenizer_clip(
            uncond_tokens,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        negative_prompt_attention_mask = uncond_input.attention_mask
        negative_prompt_attention_mask = negative_prompt_attention_mask.to(device)

        negative_prompt_embeds = pipeline.clip_l_model(uncond_input.input_ids.to(device), attention_mask=negative_prompt_attention_mask)
        negative_prompt_embeds_pooling = negative_prompt_embeds[1]
        negative_prompt_embeds = negative_prompt_embeds[0]

    if do_classifier_free_guidance:
        # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
        seq_len = negative_prompt_embeds.shape[1]

        negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype, device=device)
        negative_prompt_embeds_pooling = negative_prompt_embeds_pooling.to(dtype=dtype, device=device)

        negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        negative_prompt_embeds_pooling = torch.repeat_interleave(negative_prompt_embeds_pooling, num_images_per_prompt, dim=0)

        negative_prompt_attention_mask = negative_prompt_attention_mask.view(bs_embed, -1)
        negative_prompt_attention_mask = negative_prompt_attention_mask.repeat(num_images_per_prompt, 1)
    else:
        negative_prompt_embeds = None
        negative_prompt_attention_mask = None
        negative_prompt_embeds_pooling = None

    return (
        prompt_embeds,
        prompt_attention_mask,
        negative_prompt_embeds,
        negative_prompt_attention_mask,
        prompt_embeds_pooling,
        negative_prompt_embeds_pooling,
    )


def encode_prompt_clip_g(
    pipeline,
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
    max_length: int = 77,  # NOTE(m2v)
    **kwargs,
):

    if "mask_feature" in kwargs:
        deprecation_message = "The use of `mask_feature` is deprecated. It is no longer used in any computation and that doesn't affect the end results. It will be removed in a future version."
        deprecate("mask_feature", "1.0.0", deprecation_message, standard_warn=False)

    if device is None:
        device = pipeline._execution_device

    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    if prompt_embeds is None:
        prompt = pipeline._text_preprocessing(prompt, clean_caption=clean_caption)
        text_inputs = open_clip.tokenize(prompt)
        prompt_embeds, prompt_embeds_pooled = encode_with_transformer(pipeline.clip_g_model, text_inputs.to(device))

    if pipeline.clip_g_model is not None:
        dtype = pipeline.clip_g_model.dtype
    elif pipeline.transformer is not None:
        dtype = pipeline.transformer.dtype
    else:
        dtype = None

    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    bs_embed, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

    # get unconditional embeddings for classifier free guidance
    if do_classifier_free_guidance and negative_prompt_embeds is None:
        uncond_tokens = [negative_prompt] * batch_size
        uncond_tokens = pipeline._text_preprocessing(uncond_tokens, clean_caption=clean_caption)
        max_length = prompt_embeds.shape[1]
        uncond_input = open_clip.tokenize(uncond_tokens)
        negative_prompt_embeds, negative_prompt_embeds_pooled = encode_with_transformer(pipeline.clip_g_model, uncond_input.to(device))

    if do_classifier_free_guidance:
        # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
        seq_len = negative_prompt_embeds.shape[1]

        negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype, device=device)

        negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    else:
        negative_prompt_embeds = None
        negative_prompt_embeds_pooled = None

    return prompt_embeds, negative_prompt_embeds, prompt_embeds_pooled, negative_prompt_embeds_pooled
