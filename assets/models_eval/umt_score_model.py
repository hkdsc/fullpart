import json
import torch
import torch.nn as nn
from einops import rearrange
import numpy as np
from assets.models_eval.UMT.umt import UMT
from assets.models_eval.UMT.backbones.bert.tokenization_bert import BertTokenizer
from PIL import Image
from torchvision import transforms
from assets.models_eval.UMT.easydict import EasyDict


class UMTScoreModel(nn.Module):
    def __init__(self):
        super().__init__()
        with open('assets/models_eval/umt_config.json', 'r') as file:
            config = json.load(file)
        self.config = EasyDict(config)
        self.tokenizer = BertTokenizer.from_pretrained(self.config.model.text_encoder.pretrained)
        self.model = UMT(self.config, self.tokenizer, is_pretrain=False)
        self.load_ckpt()
        self.model.eval()

        self.transforms = transforms.Compose(
            [
                transforms.Resize(
                    (self.config.inputs.image_res, self.config.inputs.image_res),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.Lambda(lambda x: x.float().div(255.0)),
                transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),
            ]
        )

    def get_frame_indices(self, vlen):
        acc_samples = min(self.config.num_frames, vlen)
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
        return frame_indices

    def load_ckpt(self):
        checkpoint = torch.load(self.config.pretrained_path, map_location="cpu")
        if 'model' in checkpoint.keys():
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
        self.model.load_state_dict(state_dict, strict=False)

    def extract_text_feats(self, texts):
        text_input = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=128, #self.config.max_txt_l,
            return_tensors="pt",
        ).to("cuda")
        text_feats = self.model.encode_text(text_input)[0]
        text_atts = text_input.attention_mask
        return text_feats, text_atts

    def extract_vision_feats(self, images):
        image_feat, pooled_image_feat = self.model.encode_vision(images, test=True)
        if len(image_feat.shape) == 4:
            image_feat = rearrange(image_feat, "b t l c -> b (t l) c").contiguous()
        image_feat = image_feat.unsqueeze(1)
        return image_feat, pooled_image_feat

    def preprocess(self, videos):
        bs = videos.shape[0]
        frame_indices = self.get_frame_indices(vlen=videos.shape[1])
        frames = torch.from_numpy(videos[:, frame_indices])
        frames = rearrange(frames, "b f h w c -> (b f) c h w")
        frames = self.transforms(frames)
        frames = rearrange(frames, "(b f) c h w -> b f c h w", b=bs)
        return frames

    @torch.no_grad()
    def forward(self, videos, prompts):
        device = "cuda"

        text_feats, text_atts = self.extract_text_feats(prompts)
        videos = self.preprocess(videos).cuda()
        image_feats, pooled_image_feat = self.extract_vision_feats(videos)

        text_encoder = self.model.get_text_encoder()

        scores = []
        for i in range(0, len(prompts)):
            encoder_output = (
                image_feats[i:i+1, 0].to(
                    device, non_blocking=True
                )
            )
            encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(
                device, non_blocking=True
            )

            repeat_n = (
                encoder_output.shape[0]
            )

            output = text_encoder(
                encoder_embeds=text_feats[i].repeat(repeat_n, 1, 1),
                attention_mask=text_atts[i].repeat(repeat_n, 1, 1),
                encoder_hidden_states=encoder_output,
                encoder_attention_mask=encoder_att,
                return_dict=True,
                mode="fusion",
            )

            itm_embeds = output.last_hidden_state[:, 0]
            score = self.model.itm_head(itm_embeds)[:, 1]
            scores.append(score)
        scores = torch.cat(scores, dim=0)
        return scores
