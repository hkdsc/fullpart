from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.pipelines import StableDiffusionXLPipeline
from einops import rearrange
from torchmetrics import Metric as BaseMetric
from torchmetrics import PeakSignalNoiseRatio as BasePSNR
from torchmetrics import StructuralSimilarityIndexMeasure as BaseSSIM
from torchmetrics.functional.multimodal.clip_score import _clip_score_update
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity as BaseLPIPS
from torchmetrics.multimodal.clip_score import CLIPScore as BaseCLIPScore

from ..configs.base_config import InstantiateConfig, dataclass, field


class Temp(BaseMetric):
    def __init__(self):
        super().__init__()
        self.add_state("l1", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, batch, video):
        self.l1 += (batch["videos"] - video).abs().mean()
        self.total += 1

    def compute(self):
        return self.l1.float() / self.total


class SDScore(BaseCLIPScore):
    full_state_update = False

    def __init__(self, num_images_per_prompt=5, sd_ckpt_path="/group/ckpt/diffusers/stable-diffusion-xl-base-1.0", device="cuda"):
        super().__init__(model_name_or_path="openai/clip-vit-base-patch32")
        self.add_state("sd_score", torch.tensor(0.0), dist_reduce_fx="cat")
        self.num_images_per_prompt = num_images_per_prompt
        self.t2i_pipeline = StableDiffusionXLPipeline.from_pretrained(pretrained_model_name_or_path=sd_ckpt_path)
        self.t2i_pipeline.to(device)
        self.t2i_pipeline.enable_vae_slicing()

    @torch.no_grad()
    def update(self, batch, video):
        prompt = batch["prompt"]  # b
        images = self.t2i_pipeline(prompt, num_images_per_prompt=self.num_images_per_prompt, output_type="pt").images  # list[(b n) c h w] \in [0, 1]
        images = images.mul(255).byte()
        processed_input = self.processor(images=[i.cpu() for i in images], return_tensors="pt", padding=True)
        img_features = self.model.get_image_features(processed_input["pixel_values"].to(images[0].device))
        img_features = img_features / img_features.norm(p=2, dim=-1, keepdim=True)
        img_features_bnd = rearrange(img_features, "(b n) d -> b n d", n=self.num_images_per_prompt)

        # video: (b f) c h w \in [-1, 1]
        batch_size = len(batch["prompt"])
        num_frames = video.shape[0] // batch_size
        frames = video.add(1).div(2).mul(255).byte()
        frames = list(frames)
        processed_input = self.processor(images=[i.cpu() for i in frames], return_tensors="pt", padding=True)
        vid_features = self.model.get_image_features(processed_input["pixel_values"].to(frames[0].device))
        vid_features = vid_features / vid_features.norm(p=2, dim=-1, keepdim=True)
        vid_features_bfd = rearrange(vid_features, "(b f) d -> b f d", f=num_frames)

        # vid_features_bfd: b f d
        # img_features_bnd: b n d
        sd_score = torch.einsum("bfd,bnd->bfn", vid_features_bfd, img_features_bnd).mean(axis=[-2, -1])

        # sd_score: b
        self.sd_score = sd_score

    def compute(self):
        return self.sd_score


class CLIPTemp(BaseCLIPScore):
    full_state_update = False

    def __init__(self):
        super().__init__(model_name_or_path="openai/clip-vit-base-patch32")
        self.add_state("temp", torch.tensor(0.0), dist_reduce_fx="cat")

    @torch.no_grad()
    def update(self, batch, video):
        # video: (b f) c h w \in [-1, 1]
        batch_size = len(batch["prompt"])
        num_frames = video.shape[0] // batch_size

        images = video.add(1).div(2).mul(255).byte()
        images = list(images)

        processed_input = self.processor(images=[i.cpu() for i in images], return_tensors="pt", padding=True)

        img_features = self.model.get_image_features(processed_input["pixel_values"].to(images[0].device))
        img_features = img_features / img_features.norm(p=2, dim=-1, keepdim=True)

        img_features_bfchw = rearrange(img_features, "(b f) d -> b f d", f=num_frames)

        # temp: b f d -> b
        temp = (img_features_bfchw[:, 1:] * img_features_bfchw[:, :-1]).sum(axis=-1).mean(axis=-1)
        self.temp = temp
        self.n_samples += batch_size

    def compute(self):
        return self.temp


class CLIPScore(BaseCLIPScore):
    full_state_update = False

    def __init__(self):
        super().__init__(model_name_or_path="openai/clip-vit-base-patch32")

    @torch.no_grad()
    def update(self, batch, video):
        # video: (b f) c h w \in [-1, 1]
        num_frames = video.shape[0] // len(batch["prompt"])
        prompt = [item for item in batch["prompt"] for _ in range(num_frames)]
        video = (video * 0.5 + 0.5).clip(0, 1)

        def trunc(prompt):
            words = "".join(char for char in prompt if char.isalnum() or char.isspace()).split()
            return " ".join(words[:74])

        prompt = [trunc(p) for p in prompt]
        # score: (b f) c h w -> f
        score, n_samples = _clip_score_update(video, prompt, self.model, self.processor)
        # score: f -> b f
        score = score.reshape(-1, num_frames)
        self.score = score.mean(1) / 100

    def compute(self):
        return self.score


class SSIMVideo(BaseSSIM):
    full_state_update = False

    def __init__(self):
        super().__init__(data_range=2, reduction="none")

    @torch.no_grad()
    def update(self, batch, video):
        # video: (b f) c h w \in [-1, 1]
        num_frames = video.shape[0] // len(batch["prompt"])
        self.num_frames = num_frames

        # image: b c h w \in [-1, 1] -> bf c h w
        image = batch["image"]
        image_rep = []
        for img in image:
            img = img.unsqueeze(0).repeat(num_frames, 1, 1, 1)
            image_rep.append(img)
        image = torch.cat(image_rep, dim=0)

        super().update(image, video)

    def compute(self):
        # res: (b f)
        res = super().compute()
        res = rearrange(res, "(b f) -> b f", f=self.num_frames)
        res = res.mean(dim=1)
        return res


class ImageConsistency(BaseCLIPScore):
    full_state_update = False

    def __init__(self):
        super().__init__(model_name_or_path="openai/clip-vit-base-patch32")
        self.add_state("temp", torch.tensor(0.0), dist_reduce_fx="cat")

    @torch.no_grad()
    def update(self, batch, video):
        # video: (b f) c h w \in [-1, 1]
        batch_size = len(batch["prompt"])
        num_frames = video.shape[0] // batch_size

        # input_image: b c h w \in [-1, 1] -> (b f) c h w \in [0, 255]
        image = batch["image"].add(1).div(2).mul(255).byte()
        images = []
        for img in image:
            img = img.unsqueeze(0).repeat(num_frames, 1, 1, 1)
            images.extend(list(img))

        # video \in [-1,1] -> \in [0, 255]
        video = video.add(1).div(2).mul(255).byte()
        video = list(video)

        # b f d
        video_feature_bfd = self.get_img_features(video, num_frames)
        image_feature_bfd = self.get_img_features(images, num_frames)

        # temp b f d -> b
        temp = (video_feature_bfd * image_feature_bfd).sum(axis=-1).mean(axis=-1)

        self.temp = temp

    def get_img_features(self, images, num_frames):
        # images: a list , element shape:  c h w \in [-1, 1] , len: b * f

        processed_input = self.processor(images=[i.cpu() for i in images], return_tensors="pt", padding=True)

        img_features = self.model.get_image_features(processed_input["pixel_values"].to(images[0].device))
        img_features = img_features / img_features.norm(p=2, dim=-1, keepdim=True)

        img_features_bfd = rearrange(img_features, "(b f) d -> b f d", f=num_frames)

        return img_features_bfd

    def compute(self):
        return self.temp


class SSIM(BaseSSIM):
    full_state_update = False

    def __init__(self):
        super().__init__(data_range=2, reduction="none")

    @torch.no_grad()
    def update(self, batch, video):
        super().update(batch, video)

    def compute(self):
        res = super().compute()
        res = res.mean()

        return res


class PSNR(BasePSNR):
    full_state_update = False
    higher_is_better = True

    def __init__(self):
        super().__init__(data_range=2, reduction="none")

    @torch.no_grad()
    def update(self, batch, video):
        super().update(batch, video)

    def compute(self):
        res = super().compute()
        res = res.mean()
        return res


class LPIPS(BaseLPIPS):
    full_state_update = False
    higher_is_better = True

    def __init__(self):
        super().__init__(net_type="vgg", reduction="mean")

    @torch.no_grad()
    def update(self, batch, video):
        super().update(batch, video)

    def compute(self):
        res = super().compute()
        res = res.mean()
        return res


@dataclass
class MetricConfig(InstantiateConfig):
    """Configuration for Metirc instantiation"""

    _target: Type = field(default_factory=lambda: Metric)
    """target class to instantiate"""

    type_name: Optional[Tuple[str, ...]] = None
    """example: ("DOVER", "CLIPScore", "SDScore", "CLIPTemp", "FlowScore")"""


class Metric(nn.Module):
    def __init__(self, config: MetricConfig, **kwargs):
        super().__init__()
        self.config = config
        if config.type_name is not None:
            for name in config.type_name:
                if name == "DOVER":
                    tmp_kwargs = {"num_frames": kwargs["num_frames"]}
                else:
                    tmp_kwargs = {}
                setattr(self, name, eval(name)(**tmp_kwargs))

    @torch.no_grad()
    def forward(self, batch, video):
        for key in self.config.type_name:
            getattr(self, key)(batch, video)

    @torch.no_grad()
    def compute(self):
        res = {}
        for key in self.config.type_name:
            metric = getattr(self, key)
            direction = "↑" if metric.higher_is_better else "↓"
            metric_res = metric.compute()
            if type(metric_res) is dict:
                for k, v in metric_res.items():
                    res[f"{key}_{k}"] = (v, direction)
            else:
                res[key] = (metric_res, direction)
        return res

    def reset(self):
        for key in self.config.type_name:
            getattr(self, key).reset()
