import torch
import torch.nn as nn
from transformers import CLIPVisionConfig, CLIPImageProcessor
from assets.models_eval.modeling_clip import CLIPVisionModel
from PIL import Image


class IQAIGCModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_model = CLIPVisionModel.from_pretrained("/group/wangqiulin/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1")
        self.linear = nn.Linear(1024, 4096)
        self.relu = nn.ReLU()

        mm_hidden_size = 4096
        self.linear_mse = nn.Linear(mm_hidden_size, 1)
        self.processor = CLIPImageProcessor.from_json_file("assets/models_eval/preprocessor_config.json")

        self.load_ckpt()

    def load_ckpt(self):
        state_dict = torch.load("/group/wangqiulin/m2v_eval_ckpt/iq_aigc/pytorch_model.bin", map_location="cpu")
        self.load_state_dict(state_dict)

    def preprocess(self, videos):
        bs, nframes, h, w = videos.shape[0], videos.shape[1], videos.shape[2], videos.shape[3]
        videos = videos.reshape(bs * nframes, h, w, 3)
        videos = [self.processor.preprocess(Image.fromarray(videos[i]), return_tensors='pt')['pixel_values'] for i in range(bs * nframes)]
        videos = torch.cat(videos, dim=0)
        return videos

    @torch.no_grad()
    def forward(self, pixel_values):
        pixel_values = self.preprocess(pixel_values).cuda()
        output = self.vision_model(pixel_values, output_hidden_states=True)
        image_features = output.hidden_states[-2][:, 1:]
        image_features = self.linear(image_features)
        image_features = self.relu(image_features)
        query = self.linear_mse(image_features.mean(1))
        return query.squeeze(-1)
