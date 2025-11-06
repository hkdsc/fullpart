import enum
import logging
import math
import os
import os.path as osp
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
import warnings
import hashlib

import cv2
import ipdb
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import _resize_with_antialiasing as resize_antialiasing
from pynvml import *  # 给AIP观测显存用
from rich import print
from rich.console import Console
from safetensors import safe_open
from transformers.utils.constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD

CONSOLE = Console()


class NoUpdateClass(enum.Enum):
    NO_UPDATE = enum.auto()


from typing import TypeVar, Union

T = TypeVar("T")
Updatable = Union[T, None, NoUpdateClass]
NO_UPDATE = NoUpdateClass.NO_UPDATE


def set_environments():
    warnings.filterwarnings("ignore")
    warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")

    # Avoid dataset pending
    os.environ["http_proxy"] = "http://oversea-squid1.jp.txyun:11080"
    os.environ["https_proxy"] = "http://oversea-squid1.jp.txyun:11080"
    os.environ["no_proxy"] = "localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com"
    os.environ["TORCH_HOME"] = "/group/ckpt/torchhub"
    os.environ["HF_DATASETS_CACHE"] = "/video/cache/huggingface"
    os.environ["HF_DATASETS_OFFLINE"] = "1"


class Timer:
    def __init__(self):
        self.timers = {}
        self.counters = {}

    def count(self, name, count_num=1):
        if name not in self.counters:
            self.counters[name] = 0
        self.counters[name] += count_num

    def get_count(self):
        res = []
        for name in self.counters:
            res.append(f"{name} - Count: {self.counters[name]}")
        return "\n".join(res)

    def update(self, name, time):
        if name not in self.timers:
            self.timers[name] = []
        self.timers[name].append(time)

    def get_stats(self, name):
        if name in self.timers:
            times = np.array(self.timers[name])
            avg_time = np.mean(times)
            std_dev = np.std(times)
            max_time = np.max(times)
            min_time = np.min(times)
            total_time = np.sum(times)
            return avg_time, std_dev, max_time, min_time, total_time
        else:
            print(f"Timer {name} not found!")
            return None

    def remove(self, name):
        if name in self.timers:
            del self.timers[name]

    def __str__(self) -> str:
        res = []
        header = "{:<20} {:>10} {:>10} {:>10} {:>10} {:>10}".format("Name", "Avg(s)", "Std(s)", "Max(s)", "Min(s)", "Total(s)")
        res.append(header)
        for name in self.timers:
            avg_time, std_dev, max_time, min_time, total_time = self.get_stats(name)
            line = "{:<20} {:>10.2f} {:>10.2f} {:>10.2f} {:>10.2f} {:>10.2f}".format(name, avg_time, std_dev, max_time, min_time, total_time)
            res.append(line)
        return "\n".join(res)


@contextmanager
def measure_time(name="Code", run=True, synchronize=True, rank0_only=False, timer: Timer = None, verbose=True):
    if run:
        if synchronize:
            torch.cuda.synchronize()
        start_time = time.time()
        yield
        if synchronize:
            torch.cuda.synchronize()
        using_time = time.time() - start_time
        if not rank0_only or torch.distributed.get_rank() == 0:
            if verbose:
                print(f"{name} finished in {using_time}s from rank {torch.distributed.get_rank()}")
        if timer is not None:
            timer.update(name, using_time)
    else:
        yield


def log_to_rank0(message, *args, **kwargs):
    try:
        # 如果没有初始化，这里会抛出异常
        is_initialized = torch.distributed.is_initialized()
    except RuntimeError:
        is_initialized = False
    if not is_initialized or torch.distributed.get_rank() == 0:
        CONSOLE.log(message, *args, **kwargs)


def save_model_from_ds(name: str, ds_state_dict, ckpt_path, trainable_only=False, rename_func=None):
    state_dict = {}
    for key, value in ds_state_dict.items():
        if key.startswith(f"{name}.") and (not trainable_only or value.requires_grad):
            state_dict[key.replace(f"{name}.", "")] = value
    if state_dict:
        log_to_rank0(f"Saving {name} model to {ckpt_path}...")
        if rename_func is not None:
            state_dict = rename_func(state_dict)
        os.makedirs(os.path.join(ckpt_path, name), exist_ok=True)
        torch.save(state_dict, os.path.join(ckpt_path, name, "pytorch_model.ckpt"))
    else:
        log_to_rank0(f"Cannot find any {name} parameters in ds_state_dict, skipping saving {name} model to {ckpt_path}...")


def load_model(model, ckpt_path, rename_func=None):
    log_to_rank0(f"Loading model {type(model)} from checkpoint: " + ckpt_path)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    if rename_func is not None:
        state_dict = rename_func(state_dict)

    # 遍历模型的每个参数和名称
    for name, param in model.named_parameters():
        if name in state_dict:
            # 直接更新参数值
            try:
                param.data.copy_(state_dict[name])
            except RuntimeError as e:
                log_to_rank0(f"Error loading {name}: {e}")
            state_dict.pop(name)
        else:
            log_to_rank0(f"Missing in state_dict: {name}")

    # 检查模型中不需要的参数
    if len(state_dict) > 0:
        for name in state_dict:
            log_to_rank0(f"Unexpected in state_dict: {name}")
    return model


def video2gif(video_fp, fps=30):
    if os.path.exists(video_fp):
        d = os.path.split(video_fp)[0]
        fn = os.path.splitext(os.path.basename(video_fp))[0]
        with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as temp_file:
            # 获取临时文件的名称
            palette_wfp = temp_file.name
            gif_wfp = os.path.join(d, f"{fn}.gif")
            # 生成调色板
            cmd = f'ffmpeg -i {video_fp} -vf "fps={fps},scale=-1:-1:flags=lanczos,palettegen" {palette_wfp} -y'
            subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            # 使用调色板生成 GIF
            cmd = f'ffmpeg -i {video_fp} -i {palette_wfp} -filter_complex "fps={fps},scale=-1:-1:flags=lanczos[x];[x][1:v]paletteuse" {gif_wfp} -y'
            subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    else:
        print(f"video_fp: {video_fp} not exists!")


def read_aspect(img_path, model_h, model_w, divisor=64):
    img = cv2.imread(img_path)[..., ::-1]
    img_h, img_w, _ = img.shape

    H = math.sqrt(model_h * model_w * img_h / img_w)
    W = (img_w / img_h) * H

    H = int(np.round(H / divisor)) * divisor
    W = int(np.round(W / divisor)) * divisor
    img = cv2.resize(img, (W, H))
    return img


def imread_center(path, h_max, w_max):
    # 读取图像并转换颜色空间
    img = cv2.imread(path)[..., ::-1]

    # 调用 resize_and_crop 函数
    img = resize_and_crop_image(img, h_max, w_max)

    return img


def resize_and_crop_image(input_image, h_max, w_max, do_crop=True):
    H, W, C = input_image.shape
    min_edge = min(H, W)
    max_edge = max(H, W)

    # 计算缩放比例
    k = float(h_max) / min_edge if H < W else float(w_max) / min_edge
    new_H = int(np.round(H * k / 64.0)) * 64
    new_W = int(np.round(W * k / 64.0)) * 64

    # 缩放图像
    img = cv2.resize(input_image, (new_W, new_H))

    if do_crop:
        # 获取新图像尺寸
        H, W, _ = img.shape

        # 计算裁剪的起始点
        startx = W // 2 - w_max // 2
        starty = H // 2 - h_max // 2

        # 确保裁剪点不是负数
        startx = max(0, startx)
        starty = max(0, starty)

        # 根据指定的最大尺寸裁剪图像
        img = img[starty : starty + h_max, startx : startx + w_max]

    return img


def imread_resize(path, h_max, w_max):
    img = cv2.imread(path)[..., ::-1]
    img = resize_image(img, min(h_max, w_max))
    return img


def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(input_image, (W, H))
    return img


def eval_setup(config_path: Path):
    if not isinstance(config_path, Path):
        config_path = Path(config_path)
    import yaml

    from .engine.trainer import TrainerConfig

    config = yaml.load(config_path.read_text(), Loader=yaml.Loader)
    # assert isinstance(config, TrainerConfig)
    return config


def get_substatedict(submodel_name, global_state_dict):
    return {k[len(submodel_name) + 1 :]: v for k, v in global_state_dict.items() if k.startswith(submodel_name + ".")}


def video2gif(video_fp, fps=30):
    if os.path.exists(video_fp):
        d = os.path.split(video_fp)[0]
        fn = os.path.splitext(os.path.basename(video_fp))[0]
        with tempfile.TemporaryDirectory() as tmpdirname:
            palette_wfp = os.path.join(tmpdirname, "palette.png")
            gif_wfp = os.path.join(d, f"{fn}.gif")
            # 生成调色板
            cmd = f'ffmpeg -i {video_fp} -vf "fps={fps},scale=-1:-1:flags=lanczos,palettegen" {palette_wfp} -y'
            subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            # 使用调色板生成 GIF
            cmd = f'ffmpeg -i {video_fp} -i {palette_wfp} -filter_complex "fps={fps},scale=-1:-1:flags=lanczos[x];[x][1:v]paletteuse" {gif_wfp} -y'
            subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    else:
        print(f"video_fp: {video_fp} not exists!")


def convert_markup_to_ansi(markup_string: str) -> str:
    """Convert rich-style markup to ANSI sequences for command-line formatting.

    Args:
        markup_string: Text with rich-style markup.

    Returns:
        Text formatted via ANSI sequences.
    """
    with CONSOLE.capture() as out:
        CONSOLE.print(markup_string, soft_wrap=True)
    return out.get()


def gpu(idx=0):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(idx)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


@contextmanager
def try_except(info=None):
    try:
        yield
    except Exception as e:
        if info is not None:
            logging.error(info)
        logging.exception(e)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def gen_data_list(paths, output_root=None, n_split=1, prefix=""):
    paths = np.array(paths)
    data_length = len(paths)
    idx_old = []
    res = []
    for i, idx in enumerate(chunks(np.arange(data_length), round(data_length / n_split))):
        if i == n_split or len(idx) < len(idx_old):
            idx = np.concatenate([idx_old, idx])
            i -= 1
            res = res[:-1]
        idx_old = idx
        if output_root is not None:
            output_path = os.path.join(output_root, f"data_{prefix}{i}.list")
            np.savetxt(output_path, paths[idx], fmt="%s")
            res.append(output_path)
        else:
            res.append(paths[idx])

    return res


def mkdir(path, do_rm=False, exist_ok=False):
    if do_rm and os.path.exists(path):
        os.system(f"rm -rf {path}")
    os.makedirs(path, exist_ok=exist_ok)


def shell(cmd, print_info=False, return_info=False):
    if print_info:
        os.system(cmd)
    else:
        logs = os.popen(cmd).readlines()
        if return_info:
            return logs


def mp_ipdb(accelerator):
    if accelerator.is_main_process:
        ipdb.set_trace()
    accelerator.wait_for_everyone()


def tensor2video(ten, path=None, fps=8):
    video = ((ten.cpu().permute(0, 2, 3, 1) * 0.5 + 0.5) * 255).type(torch.uint8)
    if path is not None:
        torchvision.io.write_video(path, video, fps=fps)
    return video


def load_safetensors(ckpt_path):
    state_dict = {}
    with safe_open(ckpt_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)
    return state_dict


def load_deepspeed(ckpt_path):
    new_ckpt_path = f"{ckpt_path}/pytorch_model.bin"
    if not os.path.exists(new_ckpt_path):
        shell(f"python {ckpt_path}/zero_to_fp32.py {ckpt_path} {new_ckpt_path}", print_info=True)
    state_dict = torch.load(new_ckpt_path)
    return state_dict


def load_state_dict(ckpt_path):
    # for deepspeed ckpt
    if osp.exists(f"{ckpt_path}/zero_to_fp32.py"):
        state_dict = load_deepspeed(ckpt_path)
        return state_dict

    # for diffusers or civitai ckpt
    if ckpt_path.endswith(".safetensors"):
        state_dict = load_safetensors(ckpt_path)
    else:
        state_dict = torch.load(ckpt_path, map_location="cpu")

    # for civitai ckpt
    if "state_dict" in state_dict.keys():
        state_dict = state_dict["state_dict"]
    return state_dict


def convert_lora(unet, text_encoder, state_dict, LORA_PREFIX_UNET="lora_unet", LORA_PREFIX_TEXT_ENCODER="lora_te", alpha=1.0, cache_dir=None):
    print("Converting lora with alpha: ", alpha)
    visited = []

    # directly update weight in diffusers model
    for key in state_dict:
        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

        # as we have set the alpha beforehand, so just skip
        if ".alpha" in key or key in visited:
            continue

        if "text" in key:
            if text_encoder is None:
                continue
            layer_infos = key.split(".")[0].split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
            curr_layer = text_encoder
        else:
            if unet is None:
                continue
            layer_infos = key.split(".")[0].split(LORA_PREFIX_UNET + "_")[-1].split("_")
            curr_layer = unet

        # find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)

        pair_keys = []
        if "lora_down" in key:
            pair_keys.append(key.replace("lora_down", "lora_up"))
            pair_keys.append(key)
            pair_keys.append(key.replace("lora_down", "alpha").split(".weight")[0])
        else:
            pair_keys.append(key)
            pair_keys.append(key.replace("lora_up", "lora_down"))
            pair_keys.append(key.replace("lora_up", "alpha").split(".weight")[0])

        # update weight
        if len(state_dict[pair_keys[0]].shape) == 4:
            weight_up = state_dict[pair_keys[0]].squeeze(3).squeeze(2).to(torch.float32)
            weight_down = state_dict[pair_keys[1]].squeeze(3).squeeze(2).to(torch.float32)
            weight_alpha = state_dict[pair_keys[2]].to(torch.float32)
            curr_layer.weight.data += (
                alpha * torch.mm(weight_up, weight_down).unsqueeze(2).unsqueeze(3) * (weight_alpha / weight_up.shape[1] if weight_alpha else 1.0)
            )
        else:
            weight_up = state_dict[pair_keys[0]].to(torch.float32)
            weight_down = state_dict[pair_keys[1]].to(torch.float32)
            # curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down)
            weight_alpha = state_dict[pair_keys[2]].to(torch.float32)
            curr_layer.weight.data += alpha * torch.mm(weight_up, weight_down) * (weight_alpha / weight_up.shape[1] if weight_alpha else 1.0)

        # update visited list
        for item in pair_keys:
            visited.append(item)
    return unet, text_encoder


def open_file(path):
    return [p.strip() for p in open(path).readlines()]


def get_path(root, endfix="jpg|png|jpeg|bmp"):
    if osp.isfile(root):
        paths = open_file(root)
    else:
        paths = []
        if endfix is not None:
            endfix = endfix.split("|")
        root = os.path.abspath(root)
        for fpath, dirs, fs in os.walk(root, followlinks=True):
            for f in fs:
                if endfix is None or osp.splitext(f)[-1][1:].lower() in endfix:
                    if f[0] == ".":
                        continue
                    paths.append(os.path.join(fpath, f))
    return paths


def get_current_git_commit():
    commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    return commit_hash


def parse_config_string(config_string):
    """
    Parses a configuration string into a dictionary.

    Args:
    config_string (str): A multiline string containing key-value pairs.

    Returns:
    dict: A dictionary containing the parsed key-value pairs.
    """

    def parse_value(value):
        """Parses a value from string to the appropriate data type."""
        value = value.strip()
        try:
            # Try converting the string to a Python literal (e.g., number, list, etc.)
            return eval(value)
        except:
            # If eval fails, return the string as is
            return value

    # Dictionary to hold the parsed key-value pairs
    parsed_dict = {}

    # Split the input string into lines and iterate over them
    for line in config_string.strip().split("\n"):
        # Split each line into key and value at the first '='
        if "=" in line:
            key, value = line.split("=", maxsplit=1)
            parsed_dict[key.strip()] = parse_value(value)

    return parsed_dict


def hashize(input_string):
    hash_value = hashlib.md5()
    hash_value.update(input_string.encode("utf-8"))
    hash_value = hash_value.hexdigest()
    short_hex_digest = hash_value[-8:]  # 只取最后8个字符
    return short_hex_digest


def get_test_video_basename(pipeline_config, call, val_data, prompt):
    end_fix = f"seed{pipeline_config.seed}_{val_data.height}x{val_data.width}"
    for k in call:
        if k == "guidance_scale":
            end_fix += f"_cfg{call[k]}"
        elif k == "strength":
            end_fix += f"_stren{call[k]}"
        elif k == "adapter_scale":
            end_fix += f"_ip{call[k]}"
        elif k == "downgrade_scale":
            end_fix += f"_down{call[k]}"
    if prompt != "":
        words = "".join(char for char in prompt if char.isalnum() or char.isspace()).split()
        linked_words = "_".join(words[:10])
        return f"{linked_words}.{hashize(prompt)}.{end_fix}"
    else:
        return f"{hashize(str(pipeline_config))}.{end_fix}"


def prepare_image_for_clip(image_tensor):
    image = resize_antialiasing(image_tensor, (224, 224))
    image = (image + 1.0) / 2.0
    mean = torch.tensor(OPENAI_CLIP_MEAN).to(image.device, dtype=image.dtype).view(1, 3, 1, 1)
    std = torch.tensor(OPENAI_CLIP_STD).to(image.device, dtype=image.dtype).view(1, 3, 1, 1)
    image = (image - mean) / std
    return image


def pad_to_target_shape(tensor, target_shape):
    # 计算每个维度的填充量
    padding = []  # [w1, w2, h1, h2, f1, f2, c1, c2, b1, b2]
    for current, target in zip(tensor.shape, target_shape):
        # 注意pad的顺序是从最后一个维度开始的
        padding = [0, target - current] + padding

    # 填充张量
    padded_tensor = torch.nn.functional.pad(tensor, padding) # NOTE(lihe): bug here?

    # 生成填充掩码, mask.shape = [b, 1, f, h, w]
    mask = torch.ones_like(tensor[:, :1], dtype=tensor.dtype)
    padded_mask = torch.nn.functional.pad(mask, padding, value=0)
    return padded_tensor, padded_mask

def pad_to_target_shape_voxel(tensor, target_shape):
    # 计算每个维度的填充量
    padding = []  # [d1, d2, w1, w2, h1, h2, f1, f2, c1, c2, b1, b2]
    for current, target in zip(tensor.shape, target_shape):
        # 注意pad的顺序是从最后一个维度开始的
        padding.append([0, target - current])
    reverse_padding = []
    for i in range(len(padding) - 1, -1, -1):
        reverse_padding = reverse_padding + padding[i]
    padding = reverse_padding

    # 填充张量
    padded_tensor = torch.nn.functional.pad(tensor, padding) # NOTE(lihe): bug here?

    # 生成填充掩码, mask.shape = [b, 1, f, h, w]
    mask = torch.ones_like(tensor[:, :1], dtype=tensor.dtype)
    padded_mask = torch.nn.functional.pad(mask, padding, value=0)
    return padded_tensor, padded_mask


def pack_data(data):
    sizes = [t.size() for t in data]
    _, c, max_f, max_h, max_w = [max(sizes_dim) for sizes_dim in zip(*sizes)]
    res, mask = [], []
    for ten in data:
        ten, m = pad_to_target_shape(ten, [1, c, max_f, max_h, max_w])
        res.append(ten)
        mask.append(m)
    return torch.cat(res, 0), torch.cat(mask, 0)

def pack_data_voxel(data):
    sizes = [t.size() for t in data]
    _, c, max_f, max_h, max_w, max_d = [max(sizes_dim) for sizes_dim in zip(*sizes)]
    res, mask = [], []
    for ten in data:
        ten, m = pad_to_target_shape_voxel(ten, [1, c, max_f, max_h, max_w, max_d])
        res.append(ten)
        mask.append(m)
    return torch.cat(res, 0), torch.cat(mask, 0)


def is_amd():
    try:
        # 执行lspci命令并过滤VGA控制器的信息
        output = subprocess.check_output("lspci | grep 'AMD'", shell=True).decode()
        # 检查输出中是否包含AMD
        if "AMD" in output:
            return True
        else:
            return False
    except subprocess.CalledProcessError:
        print("获取显卡信息失败")
        return False


def less_than_or_equal_to(data1, data2):
    # 检查数据类型是否相同
    if type(data1) != type(data2):
        raise TypeError("data1 和 data2 类型必须相同")

    # 如果是数字，直接比较
    if isinstance(data1, (int, float)):
        return data1 <= data2

    # 如果是列表或元组，确保它们的长度相同，然后比较每个元素
    elif isinstance(data1, (list, tuple)):
        if len(data1) != len(data2):
            raise ValueError("列表或元组的长度必须相同")

        for a, b in zip(data1, data2):
            if not (a <= b):
                return False
        return True

    # 其他类型则不支持比较
    else:
        raise TypeError("less_than_or_equal_to函数不支持的数据类型")


def wrap_conv3d():
    SPLIT_SIZE = 16
    torch_conv3d = F.conv3d

    def split_conv(input, weight, *args, **kwargs):
        # print(f"{input.shape}")
        with measure_time("conv3d", False, rank0_only=True):
            out_channels, in_channels_over_groups, kT, kH, kW = weight.shape
            if in_channels_over_groups <= SPLIT_SIZE:
                return torch_conv3d(input, weight, *args, **kwargs)
            else:
                output = None
                split_inputs = torch.chunk(input, SPLIT_SIZE, dim=1)
                split_conv_weight = torch.chunk(weight, SPLIT_SIZE, dim=1)
                for i in range(len(split_inputs)):
                    if i == 0:
                        output = torch_conv3d(split_inputs[i], split_conv_weight[i], *args, **kwargs)
                        #  since bias only needs to added once, we set it to None after i==0
                        args = list(args)
                        args[0] = None
                    else:
                        output += torch_conv3d(split_inputs[i], split_conv_weight[i], *args, **kwargs)
                return output

    F.conv3d = split_conv


class TrainingStatus(enum.Enum):
    TRAINING = 1
    VALIDATING = 2
    INIT = 3


def recover_model(module: torch.nn.Module):
    for n, p in module.named_parameters():
        # assert hasattr(p, "ds_tensor"), "we cannot recover non-zero module"
        if hasattr(p, "ds_tensor"):
            # assert False
            assert p.ds_status == ZeroParamStatus.NOT_AVAILABLE
            assert p.dtype is torch.float16
            p.all_gather()
            assert p.ds_status == ZeroParamStatus.AVAILABLE


def partition_model(module: torch.nn.Module):
    for n, p in module.named_parameters():
        # assert hasattr(p, "ds_tensor"), "we cannot recover non-zero module"
        if hasattr(p, "ds_tensor"):
            # assert False
            assert p.ds_status == ZeroParamStatus.AVAILABLE
            # assert p.dtype is torch.float16
            p.partition()
            assert p.ds_status == ZeroParamStatus.NOT_AVAILABLE


def count_numel(module: torch.nn.Module):
    total_numel = 0
    for n, p in module.named_parameters():
        if hasattr(p, "partition_numel"):
            total_numel += p.partition_numel()
        else:
            total_numel += p.numel()
    return total_numel
