import math
import os
import random
import resource
import sys
from dataclasses import dataclass, field
from typing import Literal, Optional, Type
import pandas as pd
import cv2
import decord
import numpy as np
import torch
from datasets import Dataset
from transformers import set_seed
from torch.utils.data import DataLoader
from torchvision import transforms
from packaging import version as pver
from PIL import Image
# from src.data.image_transform import RandomAugment

try:
    import pyvips

    pyvips_installed = True
except:
    pyvips_installed = False

from ..configs.base_config import InstantiateConfig
from ..utils import log_to_rank0

from PIL import Image
import ast
import json
import numpy as np
import open3d as o3d
import utils3d


def voxelize_part(file):
    mesh = o3d.io.read_triangle_mesh(file) # [-1, 1]
    # normalize vertices according to bbox
    vertices = np.asarray(mesh.vertices)
    box_max = vertices.max(axis=0)
    box_min = vertices.min(axis=0)
    box = np.stack([box_min, box_max])
    center = box.mean(axis=0)

    vertices_ = vertices - center
    max_abs = np.abs(vertices_).max() * 2.
    vertices_ = vertices_ / (1e-6 + max_abs)
    # clamp vertices to the range [-0.5, 0.5]
    vertices = np.clip(np.asarray(vertices_), -0.5 + 1e-6, 0.5 - 1e-6)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(mesh, voxel_size=1/64, min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5))
    vertices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
    assert np.all(vertices >= 0) and np.all(vertices < 64), "Some vertices are out of bounds"
    vertices = (vertices + 0.5) / 64 - 0.5
    return vertices, box

def numpy_to_mp4(video_array, output_path, fps=30, is_bgr=False):
    assert len(video_array.shape) == 4, "输入必须是 (F, C, H, W) 形状"
    F, C, H, W = video_array.shape
    assert C in [1, 3], "通道数必须是 1 (灰度) 或 3 (RGB/BGR)"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H), isColor=(C==3))
    import tqdm
    for frame in tqdm.tqdm(video_array, desc="写入视频"):
        if C == 1:
            frame = frame[0]  # (H, W)
        else:
            frame = np.transpose(frame, (1, 2, 0))  # (H, W, C)
        
        if C == 3 and not is_bgr:
            frame = frame[..., ::-1]  # RGB 转 BGR

        # frame = (frame * 255).clip(0, 255).astype(np.uint8)
        frame = (frame).clip(0, 255).astype(np.uint8)
        
        video_writer.write(frame)
    
    video_writer.release()
    

def project_points_to_mask_video(points, c2w_list, intrinsics, H, W, output_path='mask_video.mp4', point_size=1):
    fx, fy, cx, cy = intrinsics
    print("===fx", fx, fy, cx, cy)
    import tqdm
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, 30.0, (W, H), False)
    
    for c2w in tqdm.tqdm(c2w_list, desc="Processing frames"):
        mask = np.zeros((H, W), dtype=np.uint8)
        w2c = np.linalg.inv(c2w)
        homogeneous_points = np.column_stack([points, np.ones(len(points))])
        camera_coords = (w2c @ homogeneous_points.T).T[:, :3]

        # valid = camera_coords[:, 2] > 0
        # camera_coords = camera_coords[valid]
        
        if len(camera_coords) == 0:
            video_writer.write(mask)
            continue

        x = camera_coords[:, 0] / camera_coords[:, 2]
        y = camera_coords[:, 1] / camera_coords[:, 2]

        u = fx * x + cx
        v = fy * y + cy

        u = np.round(u).astype(int)
        v = np.round(v).astype(int)

        valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u = u[valid]
        v = v[valid]
        
        for ui, vi in zip(u, v):
            cv2.circle(mask, (ui, vi), point_size, 255, -1)
        
        video_writer.write(mask)
    
    video_writer.release()
    print(f"Mask video saved to {output_path}")


def voxelize_point_cloud(points):
    clipped_points = np.clip(points, -0.5 + 1e-6, 0.5 - 1e-6)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(clipped_points)

    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
        pcd,
        voxel_size=1/64,
        min_bound=(-0.5, -0.5, -0.5),
        max_bound=(0.5, 0.5, 0.5)
    )
    voxels = voxel_grid.get_voxels()
    if len(voxels) == 0:
        raise ValueError("No voxels generated - input points may be empty")
    indices = np.array([voxel.grid_index for voxel in voxels])
    assert np.all(indices >= 0) and np.all(indices < 64), "Voxel indices out of bounds"
    
    vertices = (indices + 0.5) / 64 - 0.5 
    return vertices

def get_voxels(path, resolution=64):
    position = utils3d.io.read_ply(path)[0]
    return position
    # coords = ((torch.tensor(position) + 0.5) * resolution).int().contiguous()
    # ss = torch.zeros(1, resolution, resolution, resolution, dtype=torch.long)
    # ss[:, coords[:, 0], coords[:, 1], coords[:, 2]] = 1
    # return ss

def get_cond_imgs(path, total_num=24, fix_id=None):
    cond_img_list = os.listdir(path)
    inds = random.randint(0, total_num-1)
    if fix_id is not None:
        img_path = os.path.join(path, "{:03d}.png".format(fix_id))
    else:
        img_path = os.path.join(path, "{:03d}.png".format(inds))
    image = Image.open(img_path)
    if image.mode != 'RGBA' or np.array(image).shape[0] != np.array(image).shape[1]:
        print("---------------------preprocess image-------------------", path)
        image = preprocess_image(image)
        image = [image]
        assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
        image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
        image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
        image = torch.stack(image) # 1, 3, h, w
        return image
    image = [image]
    assert all(isinstance(i, Image.Image) for i in image), "Image list should be list of PIL images"
    image = [i.resize((518, 518), Image.LANCZOS) for i in image]
    image = [np.array(i.convert('RGB')).astype(np.float32) / 255 for i in image]
    image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
    image = torch.stack(image) # 1, 3, h, w
    return image

def preprocess_image(input: Image.Image) -> Image.Image:
    """
    Preprocess the input image.
    """
    # if has alpha channel, use it directly; otherwise, remove background
    has_alpha = False
    if input.mode == 'RGBA':
        alpha = np.array(input)[:, :, 3]
        if not np.all(alpha == 255):
            has_alpha = True
    if has_alpha:
        output = input
    else:
        input = input.convert('RGB')
        max_size = max(input.size)
        scale = min(1, 1024 / max_size)
        if scale < 1:
            input = input.resize((int(input.width * scale), int(input.height * scale)), Image.Resampling.LANCZOS)
        import rembg
        rembg_session = rembg.new_session('u2net')
        output = rembg.remove(input, session=rembg_session)
    output_np = np.array(output)
    alpha = output_np[:, :, 3]
    bbox = np.argwhere(alpha > 0.8 * 255)
    bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
    size = int(size * 1.2)
    # size = int(size * 1.5)
    bbox = center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2
    output = output.crop(bbox)  # type: ignore
    output = output.resize((518, 518), Image.Resampling.LANCZOS)
    output = np.array(output).astype(np.float32) / 255
    output = output[:, :, :3] * output[:, :, 3:4]
    output = Image.fromarray((output * 255).astype(np.uint8))
    return output


def calculate_K(fov, res, fov_type='horizontal'):
    W, H = res
    if fov_type == 'horizontal':
        fx = W / (2 * np.tan(fov / 2))
        fy = fx 
    elif fov_type == 'vertical':
        fy = H / (2 * np.tan(fov / 2))
        fx = fy 
    else:
        raise ValueError("fov_type error")
    cx = W / 2
    cy = H / 2
    # K = np.array([
    #     [fx, 0, cx],
    #     [0, fy, cy],
    #     [0, 0, 1]
    # ])
    K = np.array([
        [fx, fy, cx, cy]
    ], dtype=np.float32)
    return K

class Camera(object):
    def __init__(self, entry):
        self.fx, self.fy, self.cx, self.cy = entry[0:4]
        self.w2c_mat = np.eye(4)
        self.w2c_mat[:3, :] = np.array(entry[6:]).reshape(3, 4)
        self.c2w_mat = np.linalg.inv(self.w2c_mat)

def get_relative_pose(cam_params, zero_t_first_frame = True):
        abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
        abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]
        source_cam_c2w = abs_c2ws[0]
        if zero_t_first_frame:
            cam_to_origin = 0
        else:
            cam_to_origin = np.linalg.norm(source_cam_c2w[:3, 3])
        target_cam_c2w = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, -cam_to_origin],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        abs2rel = target_cam_c2w @ abs_w2cs[0]
        ret_poses = [target_cam_c2w, ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
        ret_poses = np.array(ret_poses, dtype=np.float32)
        return ret_poses


def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')

def ray_condition_voxel(voxels, c2w, device, flip_flag=None):
    # c2w: B, V, 4, 4
    # K: B, V, 4
    # voxels: N, 3, global coordinate
    B, V = c2w.shape[:2]
    N = voxels.shape[0]
    rays_o = c2w[..., :3, 3] # B, V, 3
    directions = voxels.unsqueeze(0).unsqueeze(0) - rays_o.unsqueeze(2) # [B, V, N, 3]
    rays_d = directions / directions.norm(dim=-1, keepdim=True) # [B, V, N, 3] 

    rays_o = rays_o[:, :, None].expand_as(rays_d)                   # B, V, N, 3
    # c2w @ dirctions
    rays_dxo = torch.linalg.cross(rays_o, rays_d)                          # B, V, N, 3
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], N, 6)             # B, V, N, 6
    return plucker


def ray_condition(K, c2w, H, W, device, flip_flag=None):
    # c2w: B, V, 4, 4
    # K: B, V, 4

    B, V = K.shape[:2]

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5          # [B, V, HxW]
    j = j.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5          # [B, V, HxW]

    n_flip = torch.sum(flip_flag).item() if flip_flag is not None else 0
    if n_flip > 0:
        j_flip, i_flip = custom_meshgrid(
            torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
            torch.linspace(W - 1, 0, W, device=device, dtype=c2w.dtype)
        )
        i_flip = i_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
        j_flip = j_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
        i[:, flip_flag, ...] = i_flip
        j[:, flip_flag, ...] = j_flip

    fx, fy, cx, cy = K.chunk(4, dim=-1)     # B,V, 1

    zs = torch.ones_like(i)                 # [B, V, HxW]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)              # B, V, HW, 3
    directions = directions / directions.norm(dim=-1, keepdim=True)             # B, V, HW, 3

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)        # B, V, HW, 3
    rays_o = c2w[..., :3, 3]                                        # B, V, 3
    rays_o = rays_o[:, :, None].expand_as(rays_d)                   # B, V, HW, 3
    # c2w @ dirctions
    rays_dxo = torch.linalg.cross(rays_o, rays_d)                          # B, V, HW, 3
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)             # B, V, H, W, 6
    # plucker = plucker.permute(0, 1, 4, 2, 3)
    return plucker

@dataclass
class DataConfig_3DMaster_Part(InstantiateConfig):
    """target class to instantiate"""
    _target: Type = field(default_factory=lambda: Data)

    """dataset csv file"""
    # path to dataset
    dataset_csv_path: Optional[str] = None

    # part dir
    part_dir: Optional[str] = None
    part_cond_dir: Optional[str] = None
    part_slat_dir: Optional[str] = None
    global_slat_dir: Optional[str] = None
    stage: Optional[str] = '1'
    rotate_slat: Optional[bool] = False

    """Column names"""
    # video column
    video_path_column: Optional[str] = None
    # caption column
    caption_column: Optional[str] = None
    # depth column and dropout ratio
    depth_column: Optional[str] = None
    depth_dropout: Optional[float] = None
    # IDs column and dropout ratio
    IDs_column: Optional[str] = None
    IDs_dropout: Optional[float] = None
    max_ids_num: int = 3

    # pointmap
    pointmap_column: Optional[str] = None

    # voxel
    voxel_column: Optional[str] = None

    # cond img
    cond_img_column: Optional[str] = None
    cond_img_num: Optional[int] = 24

    # part 
    part_column: Optional[str] = None

    id_mapping: Optional[bool] = False

    metadata_csv: Optional[str] = None

    global_part: Optional[bool] = False

    max_num: Optional[int] = 20

    """Parameters for processing video"""
    # whether to resize to target size at video read
    # resize_video: bool = True
    # ratio to randomly flip the video
    random_flip_ratio: float = 0.0
    # Whether to use pyvips for reading images
    use_pyvips: bool = False

    select_prompt_from_list: bool = False

    """dataloader parameters"""
    # Maximal number of samples
    num_samples: Optional[int] = None
    # Batch size
    batch_size: int = 1
    # Number of processes to use
    num_processes: int = 1
    # Whether to shuffle the dataloader
    shuffle: bool = True
    # Whether to use fully determinstic dataset
    use_determinstic_dataset: bool = False

    eval: Optional[bool] = False
    slat_norm: bool = False

class DeterminsticDataset:
    def __init__(self, data: Dataset):
        self.data = data

    def __getitem__(self, key):
        assert isinstance(key, int)
        set_seed(key + 1)
        return self.data.__getitem__(key)

    def __len__(self):
        return len(self.data)

def set_soft_limit(resource_type, soft_limit):
    try:
        current_limits = resource.getrlimit(resource_type)
        new_limits = (soft_limit, current_limits[1])
        resource.setrlimit(resource_type, new_limits)
        log_to_rank0(f"Soft limit for {resource_type} set to {soft_limit}")
    except Exception as e:
        print(f"An error occurred: {e}")

class Data:
    def __init__(self, config: DataConfig_3DMaster_Part, timer=None):
        # change the max files to open to 1048576
        set_soft_limit(resource.RLIMIT_NOFILE, 1048576)

        self.video_path_column = config.video_path_column
        self.caption_column = config.caption_column

        self.depth_column = config.depth_column
        self.depth_dropout = config.depth_dropout

        self.IDs_column = config.IDs_column
        self.max_ids_num = config.max_ids_num
        self.IDs_dropout = config.IDs_dropout

        self.voxel_column = config.voxel_column
        self.cond_img_column = config.cond_img_column
        self.cond_img_num = config.cond_img_num

        self.part_column = config.part_column
        self.part_dir = config.part_dir
        self.part_cond_dir = config.part_cond_dir
        self.part_slat_dir = config.part_slat_dir
        self.global_slat_dir = config.global_slat_dir
        self.stage = config.stage
        self.rotate_slat = config.rotate_slat

        self.id_mapping = config.id_mapping
        self.metadata_csv = config.metadata_csv
        self.global_part = config.global_part
        self.max_num = config.max_num

        self.random_flip_ratio = config.random_flip_ratio

        self.select_prompt_from_list = config.select_prompt_from_list

        self.config = config

        # cond img transforms
        cond_transform = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.image_cond_model_transform = cond_transform

        # for dataset
        if isinstance(config.dataset_csv_path, list):
            self.dataset_mirror = pd.concat([
                pd.read_csv(dataset_csv_path_i, dtype = {
                                                    self.IDs_column: object,
                                                    self.video_path_column: object,
                                                    self.caption_column: object,
                                                    self.depth_column: object,
                                                }) for dataset_csv_path_i in config.dataset_csv_path
            ])
        elif isinstance(config.dataset_csv_path, dict):
            self.dataset_mirror = pd.concat([
                pd.read_csv(dataset_csv_path_i, dtype = {
                                                    self.IDs_column: object,
                                                    self.video_path_column: object,
                                                    self.caption_column: object,
                                                    self.depth_column: object,
                                                })[:dataset_csv_path_i_num] 
                                                for dataset_csv_path_i, dataset_csv_path_i_num in config.dataset_csv_path.items()
            ])
        else:
            self.dataset_mirror = pd.read_csv(config.dataset_csv_path, 
                                                dtype = {
                                                    self.IDs_column: object,
                                                    self.video_path_column: object,
                                                    self.caption_column: object,
                                                    self.depth_column: object,
                                                })
        self.dataset = Dataset.from_pandas(self.dataset_mirror)
        self.dataset.set_transform(self.process_items)
        if config.num_samples and config.num_samples > 0:
            self.dataset = self.dataset.select(range(config.num_samples))

        # for dataloader
        if config.use_determinstic_dataset:
            # NOTE: for perfect resume, default False
            self.dataloader = DataLoader(
                DeterminsticDataset(self.dataset),
                batch_size=config.batch_size,
                collate_fn=self.collate_fn,
                num_workers=config.num_processes,
                pin_memory=True,
                shuffle=config.shuffle,
                generator=torch.Generator().manual_seed(1),
            )
        else:
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=config.batch_size,
                collate_fn=self.collate_fn,
                num_workers=config.num_processes,
                pin_memory=True,
                shuffle=config.shuffle,
            )
        
        # id mapping
        if self.id_mapping:
            df = pd.read_csv(self.metadata_csv)
            sha256_to_id = dict(zip(df['sha256'], df['file_identifier'].str.split('/').str[-1]))
            self.id_to_sha256 = {v: k for k, v in sha256_to_id.items()}
        
        slat_normalization = {
            "mean": [
                -2.1687545776367188,
                -0.004347046371549368,
                -0.13352349400520325,
                -0.08418072760105133,
                -0.5271206498146057,
                0.7238689064979553,
                -1.1414450407028198,
                1.2039363384246826
            ],
            "std": [
                2.377650737762451,
                2.386378288269043,
                2.124418020248413,
                2.1748552322387695,
                2.663944721221924,
                2.371192216873169,
                2.6217446327209473,
                2.684523105621338
            ]
        }

        self.slat_normalization = slat_normalization

        self.timer = timer
        decord.bridge.set_bridge("torch")

    def process_items(self, items):
        """ item contents """
        item_caption = items[self.caption_column] if self.caption_column in list(items.keys()) else None
        item_voxel = items[self.voxel_column] if self.voxel_column in list(items.keys()) else None
        item_cond_img = items[self.cond_img_column] if self.cond_img_column in list(items.keys()) else None
        item_part = items[self.part_column] if self.part_column in list(items.keys()) else None

        """ retuen dict """
        # voxels
        samples_voxels = []
        # samples cond imgs
        samples_cond_imgs = []
        # part
        samples_part = []
        samples_bbox = []
        samples_slat_feats = []
        samples_slat_coords = []
        # caption
        samples_caption = []
        # id
        samples_id = []
        # success
        samples_success = []

        for idx in range(len(item_part)):
            try:
                # process part
                part_id = item_part[idx]
                samples_id.append(part_id)
                info_path = os.path.join(self.part_dir, part_id, part_id + '_info.json')
                if os.path.exists(info_path):
                    with open(info_path) as f:
                        infos = json.load(f)
                    ordered_part_level = infos['ordered_part_level']
                    # bboxes = infos['bboxes']
                    num_parts = len(ordered_part_level)
                else:
                    file_list = os.listdir(os.path.join(self.part_dir, part_id))
                    file_list = [x for x in file_list if x.split('.')[-1] == 'glb']
                    num_parts = len(file_list) - 1 # has _segmented.glb
                #NOTE(lihe): only train <= 20
                num_parts = min(self.max_num, num_parts)
                # load obj and voxelize
                global_vertices_dict = {}
                local_vertices_dict = {}
                bboxes = []
                vertices_list = []

                slat_coords_list = []
                slat_feats_list = []

                # add global part
                if self.global_part:
                    global_file = os.path.join(self.part_dir, part_id, f'{part_id}_segmented.glb')
                    vertices, bbox = voxelize_part(global_file) # [-0.5, +0.5]
                    bboxes.append(bbox)
                    vertices_list.append(torch.from_numpy(vertices).float())
                    if self.stage == '2':
                        global_slat_path = os.path.join(self.global_slat_dir, 
                                                        self.id_to_sha256[part_id] + '.npz' if self.id_mapping else part_id + '.npz')
                        slat = np.load(global_slat_path)
                        slat_feats = slat['feats']
                        slat_coords = slat['coords']
                        slat_feats = torch.from_numpy(slat_feats).float()
                        slat_coords = torch.from_numpy(slat_coords).int()
                        if self.config.slat_norm:
                            # print("===normalization in dataset to N(0, I) global=====")
                            std = torch.tensor(self.slat_normalization['std'])[None].to(slat_feats.device)
                            mean = torch.tensor(self.slat_normalization['mean'])[None].to(slat_feats.device)
                            # slat_feats = slat_feats * std + mean
                            slat_feats = (slat_feats - mean) / std
                        else:
                            print("-------dont normalize--------")
                        slat_feats_list.append(slat_feats)
                        slat_coords_list.append(slat_coords)

                for i in range(num_parts):
                    # load stage2 latent
                    if self.stage == '2':
                        slat_path = os.path.join(self.part_slat_dir, part_id, f'{i}.npz')
                        if not os.path.exists(slat_path):
                            print(f"-----{i} missed, {part_id}----")
                            continue
                        try:
                            slat = np.load(slat_path)
                        except:
                            print("-----use id mapping---")
                            slat_path = os.path.join(self.part_slat_dir, self.id_to_sha256[part_id], f'{i}.npz')
                            slat = np.load(slat_path)

                        slat_feats = slat['feats']
                        slat_coords = slat['coords']
                        slat_feats = torch.from_numpy(slat_feats).float()
                        slat_coords = torch.from_numpy(slat_coords).int()
                        if self.config.slat_norm:
                            # print("===normalization in dataset to N(0, I)=====")
                            std = torch.tensor(self.slat_normalization['std'])[None].to(slat_feats.device)
                            mean = torch.tensor(self.slat_normalization['mean'])[None].to(slat_feats.device)
                            # slat_feats = slat_feats * std + mean
                            slat_feats = (slat_feats - mean) / std
                        else:
                            print("====dont normalize=====")
                        # NOTE: slat norm
                        slat_feats_list.append(slat_feats)
                        slat_coords_list.append(slat_coords)
                        
                    mesh_file = os.path.join(self.part_dir, part_id, f'{i}.glb')
                    # bbox = np.asarray(bboxes[i])
                    vertices, bbox = voxelize_part(mesh_file) # [-0.5, +0.5]
                    # local_vertices_dict[f'{i}'] = vertices
                    bboxes.append(bbox)
                    vertices_list.append(torch.from_numpy(vertices).float())
                    # recover back to global sapce, only for debugging
                    # center = bbox.mean(0)
                    # scale = (bbox[1] - bbox[0]).max()
                    # global_vertices = vertices * scale + center
                    # global_vertices_dict[f'{i}'] = global_vertices
                
                if self.stage == '2':
                    samples_slat_feats.append(slat_feats_list)
                    samples_slat_coords.append(slat_coords_list)
                
                bboxes = np.stack(bboxes) # K, 2, 3
                samples_bbox.append(torch.from_numpy(bboxes).float())
                samples_part.append(vertices_list)

                # load voxel
                if item_voxel is not None:
                    voxels = get_voxels(item_voxel[idx])
                    samples_voxels.append(torch.from_numpy(voxels).to(samples_bbox[0].dtype).to(samples_bbox[0].device))
                
                # get cond img
                if self.id_mapping:
                    try:
                        cond_img_dir = os.path.join(self.part_cond_dir, self.id_to_sha256[part_id])
                    except:
                        cond_img_dir = os.path.join(self.part_cond_dir, part_id)
                else:
                    cond_img_dir = os.path.join(self.part_cond_dir, part_id)
                fix_id=100 if self.config.eval else None
                cond_imgs = get_cond_imgs(cond_img_dir, total_num=self.cond_img_num, fix_id=fix_id)
                cond_imgs = self.image_cond_model_transform(cond_imgs).to(samples_part[0][0].dtype).to(samples_part[0][0].device)
                samples_cond_imgs.append(cond_imgs)
                
                # process caption
                if item_caption is not None:
                    caption = item_caption[idx]
                    if self.select_prompt_from_list:
                        caption_list = ast.literal_eval(caption)
                        inds = random.randint(0, len(caption_list)-1)
                        caption = caption_list[inds]

                samples_success.append(True)
            
            except Exception as e:
                print(f"Failed to process {part_id}. {e}")
                samples_caption.append(None)
                samples_id.append(None)
                samples_success.append(False)
                continue

        return_dict= {
                "samples_bbox": samples_bbox,
                "samples_part": samples_part,
                "samples_success": samples_success,
                "samples_cond_imgs": samples_cond_imgs,
                "samples_id": samples_id,
            }
        if self.stage == '2':
            return_dict["samples_slat_feats"] = samples_slat_feats
            return_dict["samples_slat_coords"] = samples_slat_coords
        return return_dict

    def collate_fn(self, examples):
        examples = [example for example in examples if example["samples_success"]]
        if self.config.batch_size > len(
            examples
        ):  # source all the required samples from the original dataset at random
            diff = self.config.batch_size - len(examples)
            count = 0
            while diff != 0:
                if count >= 30:
                    raise Exception("Encounter 10 bad samples continuously! Exit!")
                sample = self.dataset[np.random.randint(0, len(self.dataset))]
                if sample["samples_success"]==False:
                    count += 1
                    continue
                examples.append(sample)
                diff -= 1

        batch_cond_imgs = [example["samples_cond_imgs"] for example in examples]
        batch_part = [example["samples_part"] for example in examples]
        batch_bbox = [example["samples_bbox"] for example in examples]

        batch_id = [example["samples_id"] for example in examples]

        return_dict = {
                "batch_cond_imgs": batch_cond_imgs,
                "batch_part": batch_part,
                "batch_bbox": batch_bbox,
                "batch_id": batch_id,
            }

        if "samples_slat_feats" in examples[0].keys():
            batch_slat_feats = [example["samples_slat_feats"] for example in examples]
            batch_slat_coords = [example["samples_slat_coords"] for example in examples]
            return_dict["batch_slat_feats"] = batch_slat_feats
            return_dict["batch_slat_coords"] = batch_slat_coords
            
        return return_dict
