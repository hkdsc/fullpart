import torch
import torch.nn as nn
from transformers import CLIPImageProcessor
from assets.models_eval.modeling_clip import CLIPVisionModel
from PIL import Image
from einops import rearrange


class VQAIGCModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.vision_model = CLIPVisionModel.from_pretrained("/group/wangqiulin/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1")

        self.config = self.vision_model.config
        mm_hidden_size = 1024
        self.linear_mse = nn.Linear(mm_hidden_size, 1)

        self.max_frames = 16
        self.pe = nn.Parameter(torch.randn(1, self.max_frames, mm_hidden_size) * .02)
        encoder_layer = nn.TransformerEncoderLayer(d_model=mm_hidden_size, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.processor = CLIPImageProcessor.from_json_file("assets/models_eval/preprocessor_config.json")

        self.load_ckpt()

    def load_ckpt(self):
        state_dict = torch.load("/group/wangqiulin/m2v_eval_ckpt/vq_aigc/pytorch_model.bin", map_location="cpu")
        self.load_state_dict(state_dict)

    def preprocess(self, videos):
        videos = videos[:, 0:self.max_frames]
        bs, nframes, h, w = videos.shape[0], videos.shape[1], videos.shape[2], videos.shape[3]
        videos = videos.reshape(bs * nframes, h, w, 3)
        videos = [self.processor.preprocess(Image.fromarray(videos[i]), return_tensors='pt')['pixel_values'] for i in range(bs*nframes)]
        videos = torch.cat(videos, dim=0)
        videos = rearrange(videos, "(b f) c h w -> b f c h w", b=bs)
        return videos

    @torch.no_grad()
    def forward(self, pixel_values):
        pixel_values = self.preprocess(pixel_values).cuda()
        bs, nf, c, h, w = pixel_values.shape[0], pixel_values.shape[1], pixel_values.shape[2], pixel_values.shape[3], pixel_values.shape[4]
        output_ve = self.vision_model(pixel_values.view(-1, c, h, w), output_hidden_states=True)
        image_features = output_ve.hidden_states[-2][:, 1:]
        image_features = image_features.mean(1)
        image_features = image_features.view(bs, -1, image_features.shape[-1])
        image_features += self.pe.repeat(bs, 1, 1)[:, 0:nf, :]
        image_features = self.transformer_encoder(image_features)
        query = self.linear_mse(image_features.mean(1))
        return query.squeeze(-1) # B
