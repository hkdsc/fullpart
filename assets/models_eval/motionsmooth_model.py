import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
from PIL import Image
from torchvision import transforms
from assets.models_eval.amt.networks.AMT import Model


class InputPadder:
    """ Pads images such that dimensions are divisible by divisor """
    def __init__(self, ht, wd, divisor=16):
        self.ht, self.wd = ht, wd
        pad_ht = (((self.ht // divisor) + 1) * divisor - self.ht) % divisor
        pad_wd = (((self.wd // divisor) + 1) * divisor - self.wd) % divisor
        self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]

    def pad(self, *inputs):
        if len(inputs) == 1:
            return F.pad(inputs[0], self._pad, mode='replicate')
        else:
            return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, *inputs):
        if len(inputs) == 1:
            return self._unpad(inputs[0])
        else:
            return [self._unpad(x) for x in inputs]

    def _unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


class MotionSmoothModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = "cuda"
        self.initialization()
        self.model = Model(corr_radius=3, corr_lvls=4, num_flows=3)
        self.load_ckpt()
        self.model.eval()

    def initialization(self):
        self.embt = torch.tensor(1/2).float().view(1, 1, 1, 1).to(self.device)

    def load_ckpt(self):
        ckpt = torch.load("/group/wangqiulin/m2v_eval_ckpt/amt/amt-s.pth", map_location="cpu")
        self.model.load_state_dict(ckpt['state_dict'])
        self.model.to(self.device)

    def preprocess(self, videos):
        bs, nf = videos.shape[0], videos.shape[1]
        videos = rearrange(videos, "b f h w c -> (b f) c h w")
        videos = torch.from_numpy(np.float32(videos) / 255.0)
        padder = InputPadder(videos.shape[2], videos.shape[3])
        inputs = [videos[i:i+1] for i in range(videos.shape[0])]
        inputs = padder.pad(*inputs)
        inputs = [inputs[i*nf:(i+1)*nf] for i in range(bs)]
        return inputs

    @torch.no_grad()
    def forward(self, videos):
        bs, nf = videos.shape[0], videos.shape[1]
        videos = self.preprocess(videos)
        gt = [videos[i][1::2] for i in range(bs)]
        inputs = [videos[i][0::2] for i in range(bs)]

        scores = []
        for bs_idx, inputs_bs in enumerate(inputs):
            outputs_bs = []
            for in_0, in_1 in zip(inputs_bs[:-1], inputs_bs[1:]):
                in_0 = in_0.to(self.device)
                in_1 = in_1.to(self.device)
                imgt_pred = self.model(in_0, in_1, self.embt, scale_factor=1, eval=True)['imgt_pred']
                outputs_bs += [imgt_pred]
            n_out = len(outputs_bs)
            diff = torch.abs(torch.cat(gt[bs_idx], dim=0)[:n_out].cuda() - torch.cat(outputs_bs, dim=0))
            scores.append(torch.mean(1.0 - diff).unsqueeze(0))
        scores = torch.cat(scores, dim=0)
        return scores
