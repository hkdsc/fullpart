"""Standalone two-stage inRaw PNG + box pair::
    
    python inference.py \
        --stage1.transformer-ckpt /path/to/stage1/checkpoint/transformer/pytorch_model.ckpt \
        --stage2.transformer-ckpt /path/to/stage2/checkpoint/transformer/pytorch_model.ckpt \
        --raw-path /path/to/sample \
        --raw-sample-id custom_id \
        --output-dir ./outputs

The defaults mirror the personal configs inside
``src/configs/train_configs``; override paths as needed for your setup.ypoint.

This script wires together the first (coarse) and second (refine) stages of the
3D multi-part generation pipeline. It reuses the existing training configs to
instantiate pipelines, loads the specified checkpoints, prepares batches from a
CSV-backed dataset, and runs both stages sequentially for each requested
sample. Stage one exports intermediate voxel reconstructions that stage two
consumes to emit final GLB meshes.

In addition to dataset-backed evaluation, the CLI can accept *raw* PNG images
paired with precomputed bounding boxes (``*.npy``). Raw inputs are first routed
through stage one so that stage two picks up the freshly produced voxel grids.

The defaults mirror the personal configs inside
``src/configs/train_configs``; override paths as needed for your setup.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random

import torch
import tyro

import numpy as np
from PIL import Image
from torchvision import transforms

from src.configs.train_configs.personal_configs_part import personal_configs_part
from src.configs.train_configs.personal_configs_part_stage2 import (
    personal_configs_part_s2,
)
from src.data import DataConfig_3DMaster_Part
from src.data.part_data import preprocess_image
from src.utils import CONSOLE


@dataclass
class StageSettings:
    """Overrides for the stage pipelines."""

    config_key: str = "3dmaster_part"
    transformer_ckpt: Optional[str] = None
    num_inference_steps: Optional[int] = None
    guidance_scale: Optional[float] = None


@dataclass
class DataOverrides:
    """Optional overrides for the stage-one evaluation dataset."""

    dataset_csv: Optional[str] = None
    part_dir: Optional[str] = None
    part_cond_dir: Optional[str] = None
    max_num: Optional[int] = None
    cond_img_num: Optional[int] = None
    global_part: Optional[bool] = None
    id_mapping: Optional[bool] = None
    metadata_csv: Optional[str] = None


@dataclass
class Args:
    """Command-line arguments parsed by ``tyro``."""

    stage1: StageSettings = field(default_factory=StageSettings)
    stage2: StageSettings = field(
        default_factory=lambda: StageSettings(config_key="3dmaster_part_s2")
    )
    data: DataOverrides = field(default_factory=DataOverrides)
    indices: Tuple[int, ...] = ()
    raw_image: Optional[Path] = None
    raw_box: Optional[Path] = None
    raw_sample_id: Optional[str] = None
    raw_path: Optional[Path] = None
    output_dir: Path = Path("./outputs")
    stage1_subdir: str = "stage1"
    stage2_subdir: str = "stage2"
    device: str = "cuda"
    skip_stage1: bool = False
    skip_stage2: bool = False


COND_IMAGE_NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)


def _resolve_device(device_arg: str) -> torch.device:
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but no GPU is available.")
    return device


def _apply_data_overrides(
    config: DataConfig_3DMaster_Part, overrides: DataOverrides
) -> DataConfig_3DMaster_Part:
    if overrides.dataset_csv is not None:
        config.dataset_csv_path = overrides.dataset_csv
    if overrides.part_dir is not None:
        config.part_dir = overrides.part_dir
    if overrides.part_cond_dir is not None:
        config.part_cond_dir = overrides.part_cond_dir
    if overrides.max_num is not None:
        config.max_num = overrides.max_num
    if overrides.cond_img_num is not None:
        config.cond_img_num = overrides.cond_img_num
    if overrides.global_part is not None:
        config.global_part = overrides.global_part
    if overrides.id_mapping is not None:
        config.id_mapping = overrides.id_mapping
    if overrides.metadata_csv is not None:
        config.metadata_csv = overrides.metadata_csv

    # Ensure inference-friendly defaults.
    config.batch_size = 1
    config.shuffle = False
    config.use_determinstic_dataset = True
    config.eval = True
    config.stage = "1"
    return config


def _instantiate_stage1(
    stage_settings: StageSettings,
    data_overrides: DataOverrides,
    device: torch.device,
) -> Tuple[DataConfig_3DMaster_Part, torch.nn.Module]:
    if stage_settings.config_key not in personal_configs_part:
        raise KeyError(
            f"Unknown stage-one config '{stage_settings.config_key}'."
        )

    trainer_cfg = copy.deepcopy(personal_configs_part[stage_settings.config_key])

    data_cfg = copy.deepcopy(trainer_cfg.val_data)
    data_cfg = _apply_data_overrides(data_cfg, data_overrides)

    pipeline_cfg = copy.deepcopy(trainer_cfg.pipeline)
    if stage_settings.transformer_ckpt is not None:
        pipeline_cfg.transformer_config.transformer_ckpt_path = (
            stage_settings.transformer_ckpt
        )
    if stage_settings.num_inference_steps is not None:
        pipeline_cfg.call["num_inference_steps"] = (
            stage_settings.num_inference_steps
        )
    if stage_settings.guidance_scale is not None:
        pipeline_cfg.call["guidance_scale"] = stage_settings.guidance_scale

    pipeline = pipeline_cfg.from_pretrained()
    pipeline = pipeline.to(device)
    pipeline.transformer.eval()
    if hasattr(pipeline, "vae") and pipeline.vae is not None:
        pipeline.vae.eval()
    if hasattr(pipeline, "voxel_vae") and pipeline.voxel_vae is not None:
        pipeline.voxel_vae.eval()

    return data_cfg, pipeline


def _instantiate_stage2(
    stage_settings: StageSettings,
    stage1_output_dir: Path,
    stage2_output_dir: Path,
    device: torch.device,
) -> torch.nn.Module:
    if stage_settings.config_key not in personal_configs_part_s2:
        raise KeyError(
            f"Unknown stage-two config '{stage_settings.config_key}'."
        )

    trainer_cfg = copy.deepcopy(personal_configs_part_s2[stage_settings.config_key])
    pipeline_cfg = copy.deepcopy(trainer_cfg.pipeline)

    if stage_settings.transformer_ckpt is not None:
        pipeline_cfg.transformer_config.transformer_ckpt_path = (
            stage_settings.transformer_ckpt
        )
    if stage_settings.num_inference_steps is not None:
        pipeline_cfg.call["num_inference_steps"] = (
            stage_settings.num_inference_steps
        )
    if stage_settings.guidance_scale is not None:
        pipeline_cfg.call["guidance_scale"] = stage_settings.guidance_scale

    pipeline_cfg.s1_save_dir = str(stage1_output_dir)
    pipeline_cfg.call["save_dir"] = str(stage2_output_dir)

    pipeline = pipeline_cfg.from_pretrained()
    pipeline = pipeline.to(device)
    pipeline.transformer.eval()
    if hasattr(pipeline, "voxel_vae"):
        pipeline.voxel_vae.eval()

    return pipeline


def _prepare_cond_imgs(batch: Dict[str, List[torch.Tensor]], device: torch.device) -> None:
    batch["batch_cond_imgs"] = [img.to(device) for img in batch["batch_cond_imgs"]]

def _to_device_dtype(obj, device, dtype) -> None:
    if isinstance(obj, torch.Tensor):
        obj = obj.to(device)
        if obj.dtype == torch.float32:
            if dtype == 'bf16':
                return obj.bfloat16()
            elif dtype == 'fp16':
                return obj.half()
        return obj
    elif isinstance(obj, dict):
        return {k: _to_device_dtype(v, device, dtype) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_device_dtype(v, device, dtype) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(_to_device_dtype(v, device, dtype) for v in obj)
    elif isinstance(obj, set):
        return set(_to_device_dtype(v, device, dtype) for v in obj)
    else:
        return obj

def _load_cond_image(image_path: Path) -> torch.Tensor:
    image = Image.open(image_path)
    if image.mode != "RGBA" or image.size[0] != image.size[1]:
        try:
            image = preprocess_image(image)
        except Exception:
            image = image.convert("RGB").resize((518, 518), Image.Resampling.LANCZOS)
    else:
        image = image.resize((518, 518), Image.Resampling.LANCZOS)
    image = image.convert("RGB")
    image_np = np.asarray(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(image_np).permute(2, 0, 1)
    tensor = COND_IMAGE_NORMALIZE(tensor)
    return tensor.unsqueeze(0)


def _load_box_tensor(box_path: Path) -> torch.Tensor:
    box_array = np.load(box_path, allow_pickle=False)
    if box_array.ndim == 2 and box_array.shape == (2, 3):
        box_array = np.expand_dims(box_array, axis=0)
    if box_array.ndim != 3 or box_array.shape[-2:] != (2, 3):
        raise ValueError(
            f"Expected bounding box array of shape (num_parts, 2, 3), got {box_array.shape}."
        )
    return torch.from_numpy(box_array).float()


def _build_canonical_parts(num_parts: int) -> List[torch.Tensor]:
    return [torch.zeros((1, 3), dtype=torch.float32) for _ in range(num_parts)]


def _prepare_raw_batch(image_path: Path, box_path: Path, sample_id: Optional[str], raw_path: Optional[Path]) -> Dict[str, List[torch.Tensor]]:
    if raw_path is not None:
        raw_path = str(raw_path)
        image_path = Path(raw_path+".png") if image_path is None else image_path
        box_path = Path(raw_path+".npy") if box_path is None else box_path
    cond_img = _load_cond_image(image_path)
    box_tensor = _load_box_tensor(box_path)
    num_parts = box_tensor.shape[0]
    parts = _build_canonical_parts(num_parts)
    batch_id = sample_id or image_path.stem
    return {
        "batch_cond_imgs": [cond_img],
        "batch_bbox": [box_tensor],
        "batch_part": [parts],
        "batch_id": [batch_id],
    }


def _fetch_sample(
    data_module, index: int
) -> Dict[str, List[torch.Tensor]]:
    dataset = data_module.dataset
    total = len(dataset)
    if index < 0:
        index = total + index
    if index < 0 or index >= total:
        raise IndexError(f"Sample index {index} out of range [0, {total}).")
    example = dataset[index]
    batch = data_module.collate_fn([example])
    return batch


def _stage1_inference(
    pipeline: torch.nn.Module,
    batch: Dict[str, List[torch.Tensor]],
    save_dir: Path,
) -> None:
    stage_config = pipeline.pipeline_config
    steps = stage_config.call.get("num_inference_steps", 25)
    guidance = stage_config.call.get("guidance_scale", 5.0)

    pipeline(
        batch_cond_imgs=batch["batch_cond_imgs"],
        batch_bbox=batch["batch_bbox"],
        batch_part=batch["batch_part"],
        batch_id=batch["batch_id"],
        save_dir=str(save_dir),
        num_inference_steps=steps,
        guidance_scale=guidance,
    )


def _stage2_inference(
    pipeline: torch.nn.Module,
    batch: Dict[str, List[torch.Tensor]],
    save_dir: Path,
    decode_part_id: Optional[List[int]] = None
) -> None:
    stage_config = pipeline.pipeline_config
    steps = stage_config.call.get("num_inference_steps", 25)
    guidance = stage_config.call.get("guidance_scale", 5.0)

    call_kwargs = pipeline.prepare_call_kwargs_i2v_from_s1(pipeline, batch)
    # 在多个part中选择8个part进行编码
    call_kwargs.update(
        {
            "save_dir": str(save_dir),
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "decode_part_id": decode_part_id,
        }
    )
    pipeline(**call_kwargs)


def main(args: Args) -> None:
    device = _resolve_device(args.device)
    torch.set_grad_enabled(False)

    stage1_output_dir = (args.output_dir / args.stage1_subdir).absolute()
    stage2_output_dir = (args.output_dir / args.stage2_subdir).absolute()
    stage1_output_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_stage2:
        stage2_output_dir.mkdir(parents=True, exist_ok=True)

    data_cfg, stage1_pipeline = _instantiate_stage1(
        args.stage1, args.data, device
    )
    data_module = None
    if len(args.indices) > 0:
        data_module = data_cfg.setup()

    stage2_pipeline = None
    if not args.skip_stage2:
        stage2_pipeline = _instantiate_stage2(
            args.stage2, stage1_output_dir, stage2_output_dir, device
        )

    all_batches: List[Dict[str, List[torch.Tensor]]] = []
    if data_module is not None:
        for raw_index in args.indices:
            batch = _fetch_sample(data_module, raw_index)
            batch = _to_device_dtype(batch, device, 'bf16')
            all_batches.append(batch)
    if args.raw_path is not None or (args.raw_image is not None and args.raw_box is not None):
        all_batches.append(_to_device_dtype(
            _prepare_raw_batch(args.raw_image, args.raw_box, args.raw_sample_id, args.raw_path), device, 'bf16'))

    if not all_batches:
        raise ValueError("No dataset indices or raw samples provided for inference.")

    for batch in all_batches:
        
        sample_id = str(batch["batch_id"][0])
        stage1_sample_dir = stage1_output_dir / sample_id
        if not args.skip_stage1:
            CONSOLE.log(f"Running stage one for '{sample_id}'.")
            _stage1_inference(stage1_pipeline, batch, stage1_output_dir)

        if stage2_pipeline is None:
            continue

        CONSOLE.log(f"Running stage two for '{sample_id}'.")
        _stage2_inference(stage2_pipeline, batch, stage2_output_dir)


if __name__ == "__main__":
    main(tyro.cli(Args))
