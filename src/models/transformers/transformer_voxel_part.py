from einops import rearrange, repeat
from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from trellis.modules.utils import convert_module_to_f16, convert_module_to_f32
from trellis.modules.transformer import AbsolutePositionEmbedder, ModulatedTransformerCrossBlock
from trellis.modules.spatial import patchify, unpatchify


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.

        Args:
            t: a 1-D Tensor of N indices, one per batch element.
                These may be fractional.
            dim: the dimension of the output.
            max_period: controls the minimum frequency of the embeddings.

        Returns:
            an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -np.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t, dtype=None):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        if dtype is not None:
            t_freq = t_freq.to(dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class SparseStructureFlowModel(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        model_channels: int,
        cond_channels: int,
        out_channels: int,
        num_blocks: int,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        mlp_ratio: float = 4,
        patch_size: int = 2,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = True,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        global_block_id_list: Optional[List] = [],
        flat_block_id_list: Optional[List] = [],
        use_zero_linear: bool = False,
        abandon_img_cond: bool = False,
        cornner_pos_emb: bool = False,
        id_emb: bool = False,
        in_out_emb: bool = False,
        cornner_in_out: bool = False,
    ):
        super().__init__()
        self.resolution = resolution
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.patch_size = patch_size
        self.pe_mode = pe_mode
        self.use_fp16 = use_fp16
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.dtype = torch.float16 if use_fp16 else torch.float32

        self.cornner_pos_emb = cornner_pos_emb
        self.id_emb = id_emb
        self.in_out_emb = in_out_emb
        self.cornner_in_out = cornner_in_out

        if self.cornner_in_out:
            assert self.in_out_emb and self.id_emb

        self.t_embedder = TimestepEmbedder(model_channels)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, 6 * model_channels, bias=True)
            )

        if pe_mode == "ape":
            if self.id_emb:
                if not self.cornner_pos_emb:
                    pos_dim = 4 
                else:
                    pos_dim = 4*8 if not self.cornner_in_out else 5*8
            else:
                pos_dim = 3 if not self.cornner_pos_emb else 3*8
            pos_embedder = AbsolutePositionEmbedder(model_channels, pos_dim)
            self.pos_embedder = pos_embedder
            # coords = torch.meshgrid(*[torch.arange(res, device=self.device) for res in [resolution // patch_size] * 3], indexing='ij')
            # coords = torch.stack(coords, dim=-1).reshape(-1, 3)
            # pos_emb = pos_embedder(coords)
            # self.register_buffer("pos_emb", pos_emb)

        self.input_layer = nn.Linear(in_channels * patch_size**3, model_channels)

        if self.in_out_emb and not self.cornner_in_out:
            self.in_box_emb = nn.Parameter(torch.randn(2, model_channels))
            nn.init.normal_(self.in_box_emb, std=1e-6)

        self.blocks = nn.ModuleList([
            ModulatedTransformerCrossBlock(
                model_channels,
                cond_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_mode='full',
                use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"),
                share_mod=share_mod,
                qk_rms_norm=self.qk_rms_norm,
                qk_rms_norm_cross=self.qk_rms_norm_cross,
                global_block=True if i in global_block_id_list else False,
                # mm_dim=1152, # video latent dim
                # flat_block=True if i in flat_block_id_list else False,
                use_zero_linear=use_zero_linear,
                abandon_img_cond=abandon_img_cond,
            )
            for i in range(num_blocks)
        ])

        self.out_layer = nn.Linear(model_channels, out_channels * patch_size**3)

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def convert_to_fp16(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.blocks.apply(convert_module_to_f32)

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        if self.share_mod:
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        else:
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)
    
    def forward_pre(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor):
        assert [*x.shape] == [x.shape[0], self.in_channels, *[self.resolution] * 3], \
                f"Input shape mismatch, got {x.shape}, expected {[x.shape[0], self.in_channels, *[self.resolution] * 3]}"
        self.dtype = x.dtype
        h = patchify(x, self.patch_size)
        h = h.view(*h.shape[:2], -1).permute(0, 2, 1).contiguous()
        h = self.input_layer(h)
        h = h + self.pos_emb[None]
        t_emb = self.t_embedder(t, dtype=self.dtype)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = t_emb.type(self.dtype)
        h = h.type(self.dtype)
        cond = cond.type(self.dtype)
        return h, t_emb, cond

    def get_part_pos_emb(self, bboxes, all_res, device, dtype, num_parts):
        # get global coords
        coords = torch.meshgrid(*[torch.arange(res) for res in all_res], indexing='ij')
        coords = torch.stack(coords, dim=-1).reshape(-1, 3)
        # get global coords
        coords_pts = (coords + 0.5) / all_res[0] - 0.5 # [-0.5, 0.5]
        num_tokens = coords_pts.shape[0]
        coords_pts = coords_pts.to(device).unsqueeze(0) # [1, 4096, 3]

        coords_global = coords_pts * (bboxes[:, 1] - bboxes[:, 0]).max(dim=-1).values[:, None, None] + bboxes.mean(dim=1)[:, None] # (b np) 4096 3
        coords_global = (coords_global + 2) / 4.
        coords_global = torch.clamp(coords_global, min=0., max=1.)
        coords_global = (coords_global * 256).int() # (b np), 4096, 3

        # id emb need to cat box id
        if self.id_emb:
            real_bs = bboxes.shape[0] // num_parts
            parts_ids = torch.arange(num_parts).view(num_parts, 1, 1)
            parts_ids = repeat(parts_ids, 'np 1 1 -> (b np) (nt 1) 1', b=real_bs, nt=num_tokens)
            parts_ids = parts_ids.to(coords_global.device).to(coords_global.dtype)
            coords_global = torch.cat([coords_global, parts_ids], dim=-1)
            coords_global = coords_global.view(-1, 4)
        else:
            coords_global = coords_global.view(-1, 3)

        pos_emb = self.pos_embedder(coords_global)

        pos_emb = pos_emb.to(device).to(dtype)

        pos_emb = rearrange(pos_emb, "(b n) c -> b n c", n=num_tokens)

        return pos_emb
    
    def get_part_corner_pos_emb(self, bboxes, all_res, device, dtype, num_parts):
        # get global coords
        with torch.no_grad():
            coords = torch.meshgrid(*[torch.arange(res) for res in all_res], indexing='ij')
            coords = torch.stack(coords, dim=-1).reshape(-1, 3)
            num_grids = coords.shape[0]
            # get 8 cornner
            coords = coords.unsqueeze(1) # N, 1, 3
            corners = torch.tensor([[-0.5, -0.5, -0.5],
                                    [-0.5, 0.5, -0.5],
                                    [0.5, 0.5, -0.5],
                                    [0.5, -0.5, -0.5],
                                    [-0.5, -0.5, 0.5],
                                    [-0.5, 0.5, 0.5],
                                    [0.5, 0.5, 0.5],
                                    [0.5, -0.5, 0.5]])
            corners = corners.unsqueeze(0) # 1, 8, 3
            coords = coords + corners # N, 8, 3
            coords = coords.reshape([num_grids*8, 3])
            # get global coords
            coords_pts = (coords + 0.5) / all_res[0] - 0.5 # [-0.5, 0.5]
            num_tokens = num_grids
            coords_pts = coords_pts.to(device).unsqueeze(0) # [1, 4096*8, 3]

            coords_global = coords_pts * (bboxes[:, 1] - bboxes[:, 0]).max(dim=-1).values[:, None, None] + bboxes.mean(dim=1)[:, None]
            # np, 4096, 3

        # in_out_emb
        if self.in_out_emb:
            # pts: np, 4096, 3
            # box: np, 2, 3
            with torch.no_grad():
                mask1 = (coords_global >= bboxes[:, 0:1]).all(dim=-1) # np, 4096 * 8
                mask2 = (coords_global <= bboxes[:, 1:2]).all(dim=-1) # np, 4096 * 8
                in_mask = mask1 & mask2 # np, 4096 * 8
                in_mask_all = in_mask.view(in_mask.shape[0], -1, 8).all(dim=-1) # np, 4096
            if not self.cornner_in_out:
                mask_emb = torch.zeros(in_mask_all.shape[0], in_mask_all.shape[1], self.model_channels, 
                                    dtype=self.in_box_emb.dtype, device=self.in_box_emb.device)
                mask_emb = mask_emb.view(-1, self.model_channels)
                in_mask_all = in_mask_all.view(-1)
                mask_emb[in_mask_all > 0] = self.in_box_emb[0]
                mask_emb[in_mask_all == 0] = self.in_box_emb[1]
            else:
                # prepare one hot coords
                mask_coords = torch.zeros(in_mask.shape[0], in_mask.shape[1], 1, device=in_mask.device).int()
                mask_coords = mask_coords.view(-1)
                mask_coords[in_mask.view(-1) > 0] = 128 # any positive number is ok, np, 4096*8
                mask_coords = mask_coords.view(in_mask.shape[0], in_mask.shape[1], 1) # np, 4096*8, 1
                mask_emb = None
        else:
            mask_emb = None

        with torch.no_grad():
            coords_global = (coords_global + 2) / 4.
            coords_global = torch.clamp(coords_global, min=0., max=1.)
            coords_global = (coords_global * 256).int() # np, 4096, 3

            # id emb need to cat box id
            if self.id_emb:
                real_bs = bboxes.shape[0] // num_parts
                parts_ids = torch.arange(num_parts).view(num_parts, 1, 1)
                parts_ids = repeat(parts_ids, 'np 1 1 -> (b np) (nt 1) 1', b=real_bs, nt=coords_global.shape[1])
                parts_ids = parts_ids.to(coords_global.device).to(coords_global.dtype)
                if not self.cornner_in_out:
                    coords_global = torch.cat([coords_global, parts_ids], dim=-1)
                    coords_global = coords_global.view(-1, 8, 4).view(-1, 32)
                else:
                    coords_global = torch.cat([coords_global, parts_ids, mask_coords], dim=-1)
                    coords_global = coords_global.view(-1, 8, 5).view(-1, 40)
            else:
                coords_global = coords_global.view(-1, 8, 3).view(-1, 24)

            pos_emb = self.pos_embedder(coords_global)
            pos_emb = pos_emb.to(device).to(dtype)

        if mask_emb is not None:
            pos_emb = pos_emb + mask_emb

        pos_emb = rearrange(pos_emb, "(b n) c -> b n c", n=num_tokens)

        return pos_emb


    def forward_pre_part(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor, bboxes: torch.Tensor,
                         np: int):
        assert [*x.shape] == [x.shape[0], self.in_channels, *[self.resolution] * 3], \
                f"Input shape mismatch, got {x.shape}, expected {[x.shape[0], self.in_channels, *[self.resolution] * 3]}"

        assert bboxes.shape[0] == x.shape[0]
        num_parts = x.shape[0]

        if not self.cornner_pos_emb:
            pos_emb = self.get_part_pos_emb(bboxes, all_res=[self.resolution] * 3, device=x.device, dtype=x.dtype, num_parts=np)
        else:
            pos_emb = self.get_part_corner_pos_emb(bboxes, all_res=[self.resolution] * 3, device=x.device, dtype=x.dtype, num_parts=np)

        self.dtype = x.dtype
        h = patchify(x, self.patch_size)
        h = h.view(*h.shape[:2], -1).permute(0, 2, 1).contiguous()
        h = self.input_layer(h)
        h = h + pos_emb
        t_emb = self.t_embedder(t, dtype=self.dtype)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = t_emb.type(self.dtype)
        h = h.type(self.dtype)
        cond = cond.type(self.dtype)
        cond = repeat(cond, "b n c -> (b np) n c", np=np)
        return h, t_emb, cond

    def forward_post(self, h, dtype):
        h = h.type(dtype)
        h = F.layer_norm(h, h.shape[-1:])
        h = self.out_layer(h)

        h = h.permute(0, 2, 1).view(h.shape[0], h.shape[2], *[self.resolution // self.patch_size] * 3)
        h = unpatchify(h, self.patch_size).contiguous()

        return h

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        assert [*x.shape] == [x.shape[0], self.in_channels, *[self.resolution] * 3], \
                f"Input shape mismatch, got {x.shape}, expected {[x.shape[0], self.in_channels, *[self.resolution] * 3]}"
        self.dtype = x.dtype
        h = patchify(x, self.patch_size)
        h = h.view(*h.shape[:2], -1).permute(0, 2, 1).contiguous()

        h = self.input_layer(h)
        h = h + self.pos_emb[None]
        t_emb = self.t_embedder(t, dtype=self.dtype)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = t_emb.type(self.dtype)
        h = h.type(self.dtype)
        cond = cond.type(self.dtype)
        for block in self.blocks:
            h = block(h, t_emb, cond)
        h = h.type(x.dtype)
        h = F.layer_norm(h, h.shape[-1:])
        h = self.out_layer(h)

        h = h.permute(0, 2, 1).view(h.shape[0], h.shape[2], *[self.resolution // self.patch_size] * 3)
        h = unpatchify(h, self.patch_size).contiguous()

        return h
