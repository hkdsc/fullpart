import os
import random
import numpy as np


import torch
try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

import deepspeed
from .accelerate_scheduler import AcceleratedScheduler
from .accelerate_dataset import prepare_data_loader as accelerate_prepare_data_loader


SAVE_STATE_PRE_HOOK = []
LOAD_STATE_PRE_HOOK = []
MODEL_NAME = "pytorch_model"
RNG_STATE_NAME = "random_states"
SCHEDULER_NAME = "scheduler"


def prepare(*args, device_placement=None):
    """
    Prepare all objects passed in `args` for distributed training and mixed precision, then return them in the same
    order.

    Args:
        *args (list of objects):
            Any of the following type of objects:

            - `torch.utils.data.DataLoader`: PyTorch Dataloader
            - `torch.nn.Module`: PyTorch Module
            - `torch.optim.Optimizer`: PyTorch Optimizer
            - `torch.optim.lr_scheduler.LRScheduler`: PyTorch LR Scheduler

        device_placement (`list[bool]`, *optional*):
            Used to customize whether automatic device placement should be performed for each object passed. Needs
            to be a list of the same length as `args`.

    <Tip>

        You don't need to prepare a model if you only use it for inference without any kind of mixed precision

    </Tip>

    Example:

    ```python
    >>> from accelerate import Accelerator

    >>> accelerator = Accelerator()
    >>> # Assume a model, optimizer, data_loader and scheduler are defined
    >>> model, optimizer, data_loader, scheduler = accelerator.prepare(model, optimizer, data_loader, scheduler)
    ```
    """
    if device_placement is None:
        device_placement = [None for _ in args]

    result = _prepare_deepspeed(*args)

    return result if len(result) > 1 else result[0]


def prepare_data_loader(data_loader: torch.utils.data.DataLoader, device_placement=None):
    """
    Prepares a PyTorch DataLoader for training in any distributed setup. It is recommended to use
    [`Accelerator.prepare`] instead.

    Args:
        data_loader (`torch.utils.data.DataLoader`):
            A vanilla PyTorch DataLoader to prepare
        device_placement (`bool`, *optional*):
            Whether or not to place the batches on the proper device in the prepared dataloader. Will default to
            `self.device_placement`.

    Example:

    ```python
    >>> import torch
    >>> from accelerate import Accelerator

    >>> accelerator = Accelerator()
    >>> data_loader = torch.utils.data.DataLoader(...)
    >>> data_loader = accelerator.prepare_data_loader(data_loader, device_placement=True)
    ```
    """
    device_placement = True
    prepared_data_loader = accelerate_prepare_data_loader(
        data_loader,
        torch.cuda.current_device(),
        num_processes=torch.distributed.get_world_size(),
        process_index=torch.distributed.get_rank(),
        split_batches=False,
        put_on_device=device_placement,
    )
    return prepared_data_loader


def _prepare_one(obj, first_pass=False, device_placement=None):
    # First pass of preparation: DataLoader, model, optimizer
    if first_pass:
        if isinstance(obj, torch.utils.data.DataLoader):
            return prepare_data_loader(obj, device_placement=device_placement)
    return obj


def _prepare_deepspeed(*args):

    is_dataloader_present = any(isinstance(obj, torch.utils.data.DataLoader) for obj in args)
    if is_dataloader_present:
        result = [
            _prepare_one(obj, first_pass=True) if isinstance(obj, torch.utils.data.DataLoader) else obj
            for obj in args
        ]

        batch_sizes = [obj.batch_size for obj in args if hasattr(obj, "batch_size")]

    model = None
    optimizer = None
    scheduler = None
    for obj in result:
        if isinstance(obj, torch.nn.Module):
            model = obj
        elif isinstance(obj, (torch.optim.Optimizer)):
            optimizer = obj
        elif (isinstance(obj, (LRScheduler))) or (
            type(obj).__name__ in deepspeed.runtime.lr_schedules.VALID_LR_SCHEDULES
        ):
            scheduler = obj

    return tuple(result)


def set_seed(seed: int, device_specific: bool = False):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.

    Args:
        seed (`int`):
            The seed to set.
        device_specific (`bool`, *optional*, defaults to `False`):
            Whether to differ the seed on each device slightly with `self.process_index`.
    """
    if device_specific:
        seed += torch.distributed.get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_rng_state(save_dir, RANK):
    # Random number generator states
    states = {}
    states_name = f"{RNG_STATE_NAME}_{RANK}.pkl"
    states["random_state"] = random.getstate()
    states["numpy_random_seed"] = np.random.get_state()
    states["torch_manual_seed"] = torch.get_rng_state()
    states["torch_cuda_manual_seed"] = torch.cuda.get_rng_state_all()
    output_states_file = os.path.join(save_dir, states_name)
    torch.save(states, output_states_file)
    if RANK <= 0:
        print(f"Random states saved in {output_states_file}")


def load_rng_state(load_dir, RANK):
    # Random states
    try:
        states = torch.load(os.path.join(load_dir, f"{RNG_STATE_NAME}_{RANK}.pkl"))
        random.setstate(states["random_state"])
        np.random.set_state(states["numpy_random_seed"])
        torch.set_rng_state(states["torch_manual_seed"])
        torch.cuda.set_rng_state_all(states["torch_cuda_manual_seed"])
    except Exception:
        if RANK <= 0:
            print("Could not load random states")


def save_deepspeed_state(model: deepspeed.DeepSpeedEngine, lr_scheduler, save_dir: str, RANK: int = -1):
    model.save_checkpoint(save_dir, MODEL_NAME)

    scheduler_name = f"{SCHEDULER_NAME}.bin"

    if RANK <= 0:
        output_scheduler_file = os.path.join(save_dir, scheduler_name)
        torch.save(lr_scheduler.state_dict(), output_scheduler_file)
        if RANK <= 0:
            print(f"Scheduler state saved in {output_scheduler_file}")

    for hook in SAVE_STATE_PRE_HOOK:
        hook([model], [], save_dir)
    save_rng_state(save_dir, RANK)

    if RANK <= 0:
        print("All states saved successfully")


def load_deepspeed_state(model: deepspeed.DeepSpeedEngine, lr_scheduler, load_dir: str, RANK: int = -1, load_module_strict=False):
    for hook in LOAD_STATE_PRE_HOOK:
        hook([], load_dir)
    model.load_checkpoint(load_dir, MODEL_NAME, load_module_strict=load_module_strict)

    scheduler_name = f"{SCHEDULER_NAME}.bin"
    input_scheduler_file = os.path.join(load_dir, scheduler_name)
    lr_scheduler.load_state_dict(torch.load(input_scheduler_file))
    load_rng_state(load_dir, RANK)

    if RANK <= 0:
        print("All states loaded successfully")
