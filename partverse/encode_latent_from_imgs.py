import os
import sys
import json
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from torchvision import transforms
from PIL import Image
from easydict import EasyDict as edict
import pandas as pd
import open3d as o3d

sys.path.insert(0, os.path.join(os.getcwd(), "src/submodule/TRELLIS"))
import utils3d
import trellis.models as models
import trellis.modules.sparse as sp

torch.set_grad_enabled(False)

def _voxelize(part_id, output_dir):
    if part_mode:
        mesh_path = os.path.join(output_dir, renders_dir_name, part_id, 'mesh.ply')
    else:
        mesh_path = os.path.join(output_dir, renders_dir_name, part_id, 'mesh.ply')
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    vertices = np.clip(np.asarray(mesh.vertices), -0.5 + 1e-6, 0.5 - 1e-6)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        mesh,
        voxel_size=1 / 64,
        min_bound=(-0.5, -0.5, -0.5),
        max_bound=(0.5, 0.5, 0.5),
    )
    vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
    if not (np.all(vertices >= 0) and np.all(vertices < 64)):
        raise ValueError(f'Some vertices out of bounds for {part_id}')
    vertices = (vertices + 0.5) / 64 - 0.5
    if part_mode:
        utils3d.io.write_ply(os.path.join(output_dir, renders_dir_name, part_id, 'voxel.ply'), vertices)
    else:
        utils3d.io.write_ply(os.path.join(output_dir, renders_dir_name, part_id, 'voxel.ply'), vertices)
    return vertices

def _load_views(frames, part_id, opt):
    renders_root = os.path.join(opt.output_dir, renders_dir_name)
    with ThreadPoolExecutor(max_workers=16) as executor:
        def worker(view):
            image_path = os.path.join(renders_root, part_id, view['file_path'])
            try:
                image = Image.open(image_path).convert('RGBA')
            except Exception as exc:
                print(f'Failed to load {image_path}: {exc}')
                return None
            image = image.resize((518, 518), Image.Resampling.LANCZOS)
            image = np.array(image).astype(np.float32) / 255.0
            image = image[:, :, :3] * image[:, :, 3:]
            image = torch.from_numpy(image).permute(2, 0, 1).float()

            c2w = torch.tensor(view['transform_matrix'])
            c2w[:3, 1:3] *= -1
            extrinsics = torch.inverse(c2w)
            fov = torch.tensor(view['camera_angle_x'])
            intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)

            return {
                'image': image,
                'extrinsics': extrinsics,
                'intrinsics': intrinsics,
            }

        for item in executor.map(worker, frames):
            if item is not None:
                yield item


def process_metadata_objaverse_xl(metadata, instances):
    sha256_dict = {}
    for instance_id in instances:
        values = metadata.loc[metadata['file_identifier'] == f'https://sketchfab.com/3d-models/{instance_id}', 'sha256'].values
        if len(values) == 0:
            print(f'Warning: No sha256 found for instance {instance_id}')
            continue
        sha256_dict[instance_id] = values[0]
    print(f'Found {len(sha256_dict)} instances in Objaverse XL after filtering')
    return sha256_dict


def collect_part_ids(opt, id_range_l, id_range_h):
    renders_root = os.path.join(opt.output_dir, renders_dir_name)
    if opt.ins_file is None:
        ins_list = sorted(os.listdir(renders_root))[id_range_l:id_range_h]
    else:
        with open(opt.ins_file, 'r') as f:
            ins_list = sorted([line.strip() for line in f.readlines()])[id_range_l:id_range_h]

    part_ids = []
    for name in ins_list:
        if isinstance(ins_list, dict):
            name = ins_list[name]
        instance_root = os.path.join(renders_root, name)
        if not os.path.isdir(instance_root):
            continue
        if not part_mode:
            sub_path = instance_root
            if os.path.isdir(sub_path) and os.path.exists(os.path.join(sub_path, 'transforms.json')):
                part_ids.append(name)
        else:
            sub_dirs = sorted(os.listdir(instance_root))
            for sub_dir in sub_dirs:
                sub_path = os.path.join(instance_root, sub_dir)
                if os.path.isdir(sub_path) and os.path.exists(os.path.join(sub_path, 'transforms.json')):
                    part_ids.append(os.path.join(name, sub_dir))
    return part_ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Encode latents from rendered images')
    parser.add_argument('--start_idx', type=int, default=0, help='start index (inclusive)')
    parser.add_argument('--end_idx', type=int, default=None, help='end index (exclusive)')
    parser.add_argument('--mode', choices=['part', 'full'], default='part', help="mode: 'part' for part renders, 'full' for full mesh renders")
    parser.add_argument('--data_root', type=str, default='dataset/partversexl', help='root output directory')
    parser.add_argument('--ins_file', type=str, default=None, help='optional instances file (one per line)')
    parser.add_argument('--feat_model', type=str, default='dinov2_vitl14_reg', help='DINOv2 feature model name')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size for view processing')
    parser.add_argument('--enc_pretrained', type=str, default='pretrained_models/trellis/ckpts/slat_enc_swin8_B_64l8_fp16', help='path to TRELLIS encoder ckpt')
    args = parser.parse_args()

    part_mode = (args.mode == 'part')
    if part_mode:
        renders_dir_name = 'textured_part_renders'
        latents_dir_name = 'textured_part_latents'
    else:
        renders_dir_name = 'textured_mesh_renders'
        latents_dir_name = 'textured_mesh_latents'

    id_range_l = args.start_idx
    id_range_h = args.end_idx if args.end_idx is not None else sys.maxsize
    opt = edict({})
    opt.output_dir = args.data_root
    opt.ins_file = args.ins_file

    opt.feat_model = args.feat_model
    opt.batch_size = args.batch_size
    opt.enc_pretrained = args.enc_pretrained

    latent_name = f'{opt.feat_model}_{opt.enc_pretrained.split("/")[-1]}'
    latents_root = os.path.join(opt.output_dir, latents_dir_name, latent_name)
    os.makedirs(latents_root, exist_ok=True)

    dinov2_model = torch.hub.load('facebookresearch/dinov2', opt.feat_model)
    dinov2_model.eval().cuda()
    feature_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    n_patch = 518 // 14

    encoder = models.from_pretrained(opt.enc_pretrained).eval().cuda()

    part_ids = collect_part_ids(opt, id_range_l, id_range_h)
    if not part_ids:
        print('No parts found to process.')
        raise SystemExit(0)

    failures = []
    with torch.no_grad():
        for part_id in tqdm(part_ids, desc='Extracting & encoding latents'):
            latent_path = os.path.join(latents_root, f'{part_id}.npz')
            if os.path.exists(latent_path):
                continue

            try:
                transforms_path = os.path.join(opt.output_dir,
                                               renders_dir_name, part_id,
                                               'transforms.json')
                with open(transforms_path, 'r') as f:
                    transforms_meta = json.load(f)
            except Exception as exc:
                print(f'Failed to read transforms for {part_id}: {exc}')
                failures.append(part_id)
                continue

            views = []
            for datum in _load_views(transforms_meta['frames'], part_id, opt):
                datum['image'] = feature_norm(datum['image'])
                views.append(datum)
            if len(views) == 0:
                print(f'No valid views for {part_id}')
                failures.append(part_id)
                continue

            if part_mode:
                voxel_file = os.path.join(opt.output_dir,
                                          'textured_part_renders_round2',
                                          part_id, 'voxel.ply')
            else:
                voxel_file = os.path.join(opt.output_dir, renders_dir_name,
                                          part_id, 'voxel.ply')
            if os.path.exists(voxel_file):
                try:
                    positions = utils3d.io.read_ply(voxel_file)[0]
                except Exception as exc:
                    print(f'Failed to read voxel file for {part_id}, re-voxelizing')
                    positions = _voxelize(part_id, opt.output_dir)
                    print(f'Re-voxelization successful for {part_id}')
            else:
                positions = _voxelize(part_id, opt.output_dir)

            positions = torch.from_numpy(positions).float().cuda()
            if positions.numel() == 0:
                print(f'Empty voxel positions for {part_id}')
                failures.append(part_id)
                continue

            indices = ((positions + 0.5) * 64).long()
            if not (torch.all(indices >= 0) and torch.all(indices < 64)):
                print(f'Invalid voxel indices for {part_id}')
                failures.append(part_id)
                continue
            indices = indices.int()

            agg_feats = torch.zeros(indices.shape[0],
                                    1024,
                                    device=positions.device)
            total_views = 0

            for start in range(0, len(views), opt.batch_size):
                batch = views[start:start + opt.batch_size]
                imgs = torch.stack([item['image'] for item in batch]).cuda()
                extr = torch.stack([item['extrinsics']
                                    for item in batch]).cuda()
                intr = torch.stack([item['intrinsics']
                                    for item in batch]).cuda()

                feats = dinov2_model(imgs, is_training=True)
                uv = utils3d.torch.project_cv(positions, extr, intr)[0] * 2 - 1
                patchtokens = feats['x_prenorm'][:, dinov2_model.
                                                 num_register_tokens + 1:]
                patchtokens = patchtokens.permute(0, 2, 1).reshape(
                    imgs.shape[0], 1024, n_patch, n_patch)
                sampled = F.grid_sample(
                    patchtokens,
                    uv.unsqueeze(1),
                    mode='bilinear',
                    align_corners=False,
                ).squeeze(2).permute(0, 2, 1)
                agg_feats += sampled.sum(dim=0)
                total_views += imgs.shape[0]

            if total_views == 0:
                print(f'No features aggregated for {part_id}')
                failures.append(part_id)
                continue

            voxel_feats = agg_feats / total_views
            tensor_coords = torch.cat([
                torch.zeros(indices.shape[0],
                            1,
                            device=indices.device,
                            dtype=torch.int32), indices
            ],
                                      dim=1)
            sparse_tensor = sp.SparseTensor(feats=voxel_feats.float(),
                                            coords=tensor_coords)
            latent = encoder(sparse_tensor, sample_posterior=False)
            if not torch.isfinite(latent.feats).all():
                print(f'Non-finite latent for {part_id}')
                failures.append(part_id)
                continue

            save_path = os.path.join(latents_root, f'{part_id}.npz')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.savez_compressed(
                save_path,
                feats=latent.feats.cpu().numpy().astype(np.float32),
                coords=latent.coords[:, 1:].cpu().numpy().astype(np.uint8),
            )

    if failures:
        print(f'Failed to process {len(failures)} parts. See list below:')
        for item in failures:
            print(item)
