import sys
from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from trellis.modules.utils import zero_module, convert_module_to_f16, convert_module_to_f32
from trellis.modules.transformer import AbsolutePositionEmbedder
from trellis.modules.norm import LayerNorm32
from trellis.modules import sparse as sp
from trellis.modules.sparse.transformer import ModulatedSparseTransformerCrossBlock
# from .sparse_structure_flow import TimestepEmbedder
from .transformer_voxel_part import TimestepEmbedder


class SparseResBlock3d(nn.Module):
    def __init__(
        self,
        channels: int,
        emb_channels: int,
        out_channels: Optional[int] = None,
        downsample: bool = False,
        upsample: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.out_channels = out_channels or channels
        self.downsample = downsample
        self.upsample = upsample
        
        assert not (downsample and upsample), "Cannot downsample and upsample at the same time"

        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm2 = LayerNorm32(self.out_channels, elementwise_affine=False, eps=1e-6)
        self.conv1 = sp.SparseConv3d(channels, self.out_channels, 3)
        self.conv2 = zero_module(sp.SparseConv3d(self.out_channels, self.out_channels, 3))
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, 2 * self.out_channels, bias=True),
        )
        self.skip_connection = sp.SparseLinear(channels, self.out_channels) if channels != self.out_channels else nn.Identity()
        self.updown = None
        if self.downsample:
            self.updown = sp.SparseDownsample(2)
        elif self.upsample:
            self.updown = sp.SparseUpsample(2)

    def _updown(self, x: sp.SparseTensor) -> sp.SparseTensor:
        if self.updown is not None:
            x = self.updown(x)
        return x

    def forward(self, x: sp.SparseTensor, emb: torch.Tensor) -> sp.SparseTensor:
        emb_out = self.emb_layers(emb).type(x.dtype)
        scale, shift = torch.chunk(emb_out, 2, dim=1)

        x = self._updown(x)
        h = x.replace(self.norm1(x.feats))
        h = h.replace(F.silu(h.feats))
        h = self.conv1(h)
        h = h.replace(self.norm2(h.feats)) * (1 + scale) + shift
        h = h.replace(F.silu(h.feats))
        h = self.conv2(h)
        h = h + self.skip_connection(x)

        return h

    def forward_split(self, x: sp.SparseTensor, emb: torch.Tensor) -> sp.SparseTensor:
        emb_out = self.emb_layers(emb).type(x.dtype)
        scale, shift = torch.chunk(emb_out, 2, dim=1)

        x = self._updown(x)
        h = x.replace(self.norm1(x.feats))
        h = h.replace(F.silu(h.feats))
        ##NOTE(lihe): split batch
        batch_size = h.coords[:, 0].max() + 1
        assert batch_size == h.shape[0]
        h_list = []
        for i in range(batch_size):
            h_list.append(self.conv1(h[i]))
        h = sp.sparse_cat(h_list)
        # h = self.conv1(h)
        ##
        h = h.replace(self.norm2(h.feats)) * (1 + scale) + shift
        h = h.replace(F.silu(h.feats))
        ##
        h_list = []
        for i in range(batch_size):
            h_list.append(self.conv2(h[i]))
        h = sp.sparse_cat(h_list)
        # h = self.conv2(h)
        ##
        h = h + self.skip_connection(x)

        return h
    

class SLatFlowModel(nn.Module):
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
        num_io_res_blocks: int = 2,
        io_block_channels: List[int] = None,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_fp16: bool = False,
        use_checkpoint: bool = False,
        use_skip_connection: bool = True,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        global_block_id_list: Optional[List] = [],
        cornner_pos_emb: bool = False,
        id_emb: bool = False,
        rotate_slat: bool = False,
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
        self.num_io_res_blocks = num_io_res_blocks
        self.io_block_channels = io_block_channels
        self.pe_mode = pe_mode
        self.use_fp16 = use_fp16
        self.use_checkpoint = use_checkpoint
        self.use_skip_connection = use_skip_connection
        self.share_mod = share_mod
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.global_block_id_list = global_block_id_list

        self.cornner_pos_emb = cornner_pos_emb
        self.id_emb = id_emb
        self.rotate_slat = rotate_slat

        #NOTE(lihe)
        self.use_checkpoint = True

        assert int(np.log2(patch_size)) == np.log2(patch_size), "Patch size must be a power of 2"
        assert np.log2(patch_size) == len(io_block_channels), "Number of IO ResBlocks must match the number of stages"

        self.t_embedder = TimestepEmbedder(model_channels)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(model_channels, 6 * model_channels, bias=True)
            )

        if pe_mode == "ape":
            if self.id_emb:
                assert self.cornner_pos_emb
                if not self.cornner_pos_emb:
                    pos_dim = 4 
                else:
                    pos_dim = 4*8
            else:
                pos_dim = 3 if not self.cornner_pos_emb else 3*8
            pos_embedder = AbsolutePositionEmbedder(model_channels, pos_dim)
            self.pos_embedder = pos_embedder
            # self.pos_embedder = AbsolutePositionEmbedder(model_channels)

        self.input_layer = sp.SparseLinear(in_channels, io_block_channels[0])
        self.input_blocks = nn.ModuleList([])
        for chs, next_chs in zip(io_block_channels, io_block_channels[1:] + [model_channels]):
            self.input_blocks.extend([
                SparseResBlock3d(
                    chs,
                    model_channels,
                    out_channels=chs,
                )
                for _ in range(num_io_res_blocks-1)
            ])
            self.input_blocks.append(
                SparseResBlock3d(
                    chs,
                    model_channels,
                    out_channels=next_chs,
                    downsample=True,
                )
            )
            
        self.blocks = nn.ModuleList([
            ModulatedSparseTransformerCrossBlock(
                model_channels,
                cond_channels,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                attn_mode='full',
                use_checkpoint=self.use_checkpoint,
                use_rope=(pe_mode == "rope"),
                share_mod=self.share_mod,
                qk_rms_norm=self.qk_rms_norm,
                qk_rms_norm_cross=self.qk_rms_norm_cross,
                global_block=True if i in global_block_id_list else False,
            )
            for i in range(num_blocks)
        ])

        self.out_blocks = nn.ModuleList([])
        for chs, prev_chs in zip(reversed(io_block_channels), [model_channels] + list(reversed(io_block_channels[1:]))):
            self.out_blocks.append(
                SparseResBlock3d(
                    prev_chs * 2 if self.use_skip_connection else prev_chs,
                    model_channels,
                    out_channels=chs,
                    upsample=True,
                )
            )
            self.out_blocks.extend([
                SparseResBlock3d(
                    chs * 2 if self.use_skip_connection else chs,
                    model_channels,
                    out_channels=chs,
                )
                for _ in range(num_io_res_blocks-1)
            ])
        self.out_layer = sp.SparseLinear(io_block_channels[0], out_channels)

        self.initialize_weights()
        if use_fp16:
            self.convert_to_fp16()

        # convert_dtype = torch.float16
        self.convert_dtype = torch.float32

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
        self.input_blocks.apply(convert_module_to_f16)
        self.blocks.apply(convert_module_to_f16)
        self.out_blocks.apply(convert_module_to_f16)
    
    def convert_to_fp16_spconv(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        # self.blocks.apply(convert_module_to_f16)
        self.out_blocks.apply(convert_module_to_f16)
    
    def convert_to_fp32_spconv(self) -> None:
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f32)
        # self.blocks.apply(convert_module_to_f16)
        self.out_blocks.apply(convert_module_to_f32)

    def convert_to_fp32(self) -> None:
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.blocks.apply(convert_module_to_f32)
        self.out_blocks.apply(convert_module_to_f32)

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
    
    def forward_pre_part(self, x: sp.SparseTensor, t: torch.Tensor, cond: torch.Tensor, bboxes, num_parts, batch_size,
                         split=False):
        self.dtype = self.input_layer.weight.dtype
        x = x.type(self.dtype)
        h = self.input_layer(x).type(self.dtype)
        # t_emb = self.t_embedder(t)
        t_emb = self.t_embedder(t.type(self.dtype), dtype=self.dtype)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = t_emb.type(self.dtype)


        if self.input_blocks[0].emb_layers[1].weight.dtype != self.convert_dtype:
            # print("====convert spconv to fp32====")
            # self.convert_to_fp16_spconv()
            self.convert_to_fp32_spconv()
        
        h = h.to(self.convert_dtype)
        t_emb = t_emb.to(self.convert_dtype)

        skips = []
        # pack with input blocks
        for block in self.input_blocks:
            h = block(h, t_emb)
            if not split:
                skips.append(h.feats)
            else:
                skips.append(h)
        
        if self.pe_mode == "ape":
            if self.cornner_pos_emb:
                all_res = [32, 32, 32] # NOTE(lihe): h is downsampled 2x
                h = h + self.get_part_corner_pos_emb(bboxes, all_res, h.coords, h.device, h.feats.dtype, num_parts, batch_size)
            else:
                raise NotImplementedError
                # h = h + self.pos_embedder(h.coords[:, 1:]).type(self.dtype)
        # return h, t_emb, cond, skips
        return h.type(self.dtype), t_emb.type(self.dtype), cond.type(self.dtype), skips
    
    def forward_post(self, h, t_emb, skips, dtype):
        convert_dtype = self.convert_dtype
        h = h.to(convert_dtype)
        t_emb = t_emb.to(convert_dtype)
        # unpack with output blocks
        for block, skip in zip(self.out_blocks, reversed(skips)):
            if self.use_skip_connection:
                h = block(h.replace(torch.cat([h.feats, skip], dim=1)), t_emb)
            else:
                h = block(h, t_emb)

        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.out_layer(h.type(dtype))
        return h
    
    def forward_post_split(self, h, t_emb, skips, dtype):
        convert_dtype = self.convert_dtype
        h = h.to(convert_dtype)
        t_emb = t_emb.to(convert_dtype)
        # unpack with output blocks
        for block, skip in zip(self.out_blocks, reversed(skips)):
            if self.use_skip_connection:
                h = block.forward_split(h.replace(torch.cat([h.feats, skip], dim=1)), t_emb)
            else:
                # h = block(h, t_emb)
                raise NotImplementedError

        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.out_layer(h.type(dtype))
        return h
    
    # def forward_post_split(self, h, t_emb, skips, dtype):
    #     convert_dtype = self.convert_dtype
    #     h = h.to(convert_dtype)
    #     t_emb = t_emb.to(convert_dtype)
    #     batch_size = h.coords[:, 0].max() + 1
    #     assert batch_size == h.shape[0]
    #     # unpack with output blocks
    #     h_list = []
    #     for i in range(batch_size):
    #         h_split = h[i]
    #         skips_split = [x[i] for x in skips]
    #         t_emb_split = t_emb[i:i+1]
    #         mask_ids = h.coords[:, 0] == i
    #         for block, skip in zip(self.out_blocks, reversed(skips_split)):
    #             if self.use_skip_connection:
    #                 h_split = block(h_split.replace(torch.cat([h_split.feats, skip.feats], dim=1)), t_emb_split)
    #             else:
    #                 h_split = block(h_split, t_emb_split)
    #         h_list.append(h_split)
        
    #     h = sp.sparse_cat(h_list)

    #     h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
    #     h = self.out_layer(h.type(dtype))
    #     return h

    def forward(self, x: sp.SparseTensor, t: torch.Tensor, cond: torch.Tensor) -> sp.SparseTensor:
        h = self.input_layer(x).type(self.dtype)
        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)
        t_emb = t_emb.type(self.dtype)
        cond = cond.type(self.dtype)

        skips = []
        # pack with input blocks
        for block in self.input_blocks:
            h = block(h, t_emb)
            skips.append(h.feats)
        
        if self.pe_mode == "ape":
            h = h + self.pos_embedder(h.coords[:, 1:]).type(self.dtype)
        for block in self.blocks:
            h = block(h, t_emb, cond)

        # unpack with output blocks
        for block, skip in zip(self.out_blocks, reversed(skips)):
            if self.use_skip_connection:
                h = block(h.replace(torch.cat([h.feats, skip], dim=1)), t_emb)
            else:
                h = block(h, t_emb)

        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.out_layer(h.type(x.dtype))
        return h
    
    def get_part_corner_pos_emb(self, bboxes, all_res, coords, device, dtype, num_parts, batch_size):
        # print("====boxes in corner pos emb====", bboxes.shape) # (b np) 2 3
        # get global coords

        corners = torch.tensor([[-0.5, -0.5, -0.5],
                                 [-0.5, 0.5, -0.5],
                                 [0.5, 0.5, -0.5],
                                 [0.5, -0.5, -0.5],
                                 [-0.5, -0.5, 0.5],
                                 [-0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5],
                                 [0.5, -0.5, 0.5]])
        corners = corners.unsqueeze(0).to(coords.device) # 1, 8, 3

        assert len(bboxes) == num_parts * batch_size
        coords_part_global_list = []
        for i in range(len(bboxes)):
            coords_part = coords[:, 1:][coords[:, 0] == i] # n x 3
            coords_part = coords_part.unsqueeze(1) # n, 1, 3
            coords_part = coords_part + corners # n, 8, 3
            coords_part = coords_part.reshape([-1, 3]) # nx8, 3
            coords_part_pts = (coords_part + 0.5) / all_res[0] - 0.5

            if self.rotate_slat:
                # print("===rotate slat in pos emb===")
                matrix = np.eye(3)
                matrix[1,1] = 0
                matrix[1,2] = 1
                matrix[2,2] = 0
                matrix[2,1] = -1
                matrix = torch.from_numpy(matrix).to(coords_part_pts.device).to(coords_part_pts.dtype)
                coords_part_pts = torch.matmul(coords_part_pts, matrix.T)[:, :3]

            coords_part_pts = coords_part_pts.to(device)
            coords_part_global = coords_part_pts * (bboxes[i, 1] - bboxes[i, 0]).max(dim=-1).values + bboxes[i].mean(dim=0)
            coords_part_global = (coords_part_global + 2) / 4.
            if not (coords_part_global.min() >= 0. and coords_part_global.max() <= 1):
                print("==========find coords exceed [0, 1]===========", coords_part_global.min(), coords_part_global.max())
            coords_part_global = torch.clamp(coords_part_global, min=0., max=1.)
            coords_part_global = (coords_part_global * 256).int() # nx8, 3

            # add id emb
            if self.id_emb:
                parts_ids = torch.zeros(coords_part_global.shape[0], 1, device=coords_part_global.device, dtype=coords_part_global.dtype) + i % num_parts
                coords_part_global = torch.cat([coords_part_global, parts_ids], dim=-1)
                coords_part_global = coords_part_global.view(-1, 8, 4).view(-1, 32)
            else:
                # no id emb now
                coords_part_global = coords_part_global.view(-1, 8, 3).view(-1, 24)
            coords_part_global_list.append(coords_part_global)

        coords_global = torch.cat(coords_part_global_list) # n_all, 24

        pos_emb = self.pos_embedder(coords_global)

        pos_emb = pos_emb.to(device).to(dtype)

        return pos_emb
