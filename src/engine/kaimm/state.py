"""
modified from https://github.com/huggingface/accelerate/blob/main/src/accelerate/state.py#L96
"""
import os
import torch
from contextlib import contextmanager
from typing import Any, Callable
from deepspeed import DeepSpeedEngine


def do_nothing(*args, **kwargs):
    return None


def extract_model_from_parallel(model, keep_fp32_wrapper: bool = True):
    """
    https://github.com/huggingface/accelerate/blob/b42c65b729209e47463754fcb271b4c3d3f964c5/src/accelerate/utils/other.py#L46
    """
    options = (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel)
    # if is_deepspeed_available():
    options += (DeepSpeedEngine,)

    while isinstance(model, options):
        model = model.module

    assert keep_fp32_wrapper

    return model


class PartialState:

    _shared_state = {}

    def __init__(self):
        self.__dict__ = self._shared_state
        if not self.initialized:
            assert torch.distributed.is_initialized(), "PartialState could work only after distributed environment is initialized."
            self.num_processes = torch.distributed.get_world_size()
            self.process_index = torch.distributed.get_rank()
            self.local_process_index = int(os.environ.get("LOCAL_RANK", -1))
            if 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ:
                self.local_process_index = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
            self.device = torch.device("cuda", self.local_process_index)
            torch.cuda.set_device(self.device)

    def __repr__(self) -> str:
        return (
            # f"Num processes: {self.num_processes}\n"
            # f"Process index: {self.process_index}\n"
            # f"Local process index: {self.local_process_index}\n"
            # f"Device: {self.device}\n"
            f"shared_state: {self._shared_state}"
        )

    @staticmethod
    def _reset_state():
        "Resets `_shared_state`, is used internally and should not be called"
        PartialState._shared_state = {}

    @property
    def initialized(self) -> bool:
        "Returns whether the `PartialState` has been initialized"
        return self._shared_state != {}

    @property
    def use_distributed(self):
        """
        Whether the Accelerator is configured for distributed training
        """
        return self.num_processes > 1

    @property
    def is_last_process(self) -> bool:
        "Returns whether the current process is the last one"
        return self.process_index == self.num_processes - 1

    @property
    def is_main_process(self) -> bool:
        "Returns whether the current process is the main process"
        return (
            self.process_index == 0     # if self.distributed_type != DistributedType.MEGATRON_LM else self.is_last_process
        )

    @property
    def is_local_main_process(self) -> bool:
        "Returns whether the current process is the main process on the local node"
        return (
            self.local_process_index == 0
            # if self.distributed_type != DistributedType.MEGATRON_LM
            # else self.is_last_process
        )

    def wait_for_everyone(self):
        if self.num_processes > 1:
            torch.distributed.barrier()

    def _goes_first(self, is_main: bool):
        if not is_main:
            self.wait_for_everyone()

        yield

        if is_main:
            self.wait_for_everyone()

    @contextmanager
    def main_process_first(self):
        """
        Lets the main process go first inside a with block.

        The other processes will enter the with block after the main process exits.

        Example:

        ```python
        >>> from accelerate import Accelerator

        >>> accelerator = Accelerator()
        >>> with accelerator.main_process_first():
        ...     # This will be printed first by process 0 then in a seemingly
        ...     # random order by the other processes.
        ...     print(f"This will be printed by process {accelerator.process_index}")
        ```
        """
        yield from self._goes_first(self.is_main_process)

    @contextmanager
    def local_main_process_first(self):
        """
        Lets the local main process go inside a with block.

        The other processes will enter the with block after the main process exits.

        Example:

        ```python
        >>> from accelerate.state import PartialState

        >>> state = PartialState()
        >>> with state.local_main_process_first():
        ...     # This will be printed first by local process 0 then in a seemingly
        ...     # random order by the other processes.
        ...     print(f"This will be printed by process {state.local_process_index}")
        ```
        """
        yield from self._goes_first(self.is_local_main_process)

    def on_main_process(self, function: Callable[..., Any] = None):
        """
        Decorator that only runs the decorated function on the main process.

        Args:
            function (`Callable`): The function to decorate.

        Example:

        ```python
        >>> from accelerate.state import PartialState

        >>> state = PartialState()


        >>> @state.on_main_process
        ... def print_something():
        ...     print("This will be printed by process 0 only.")


        >>> print_something()
        "This will be printed by process 0 only"
        ```
        """
        if not self.initialized:
            raise ValueError("The `PartialState` or `Accelerator` must be initialized before calling this function.")
        if self.is_main_process or not self.use_distributed:
            return function
        return do_nothing

    def on_local_main_process(self, function: Callable[..., Any] = None):
        """
        Decorator that only runs the decorated function on the local main process.

        Args:
            function (`Callable`): The function to decorate.

        Example:
        ```python
        # Assume we have 2 servers with 4 processes each.
        from accelerate.state import PartialState

        state = PartialState()


        @state.on_local_main_process
        def print_something():
            print("This will be printed by process 0 only on each server.")


        print_something()
        # On server 1:
        "This will be printed by process 0 only"
        # On server 2:
        "This will be printed by process 0 only"
        ```
        """
        if self.is_local_main_process or not self.use_distributed:
            return function
        return do_nothing

    def on_last_process(self, function: Callable[..., Any]):
        """
        Decorator that only runs the decorated function on the last process.

        Args:
            function (`Callable`): The function to decorate.

        Example:
        ```python
        # Assume we have 4 processes.
        from accelerate.state import PartialState

        state = PartialState()


        @state.on_last_process
        def print_something():
            print(f"Printed on process {state.process_index}")


        print_something()
        "Printed on process 3"
        ```
        """
        if self.is_last_process or not self.use_distributed:
            return function
        return do_nothing

    def on_process(self, function: Callable[..., Any] = None, process_index: int = None):
        """
        Decorator that only runs the decorated function on the process with the given index.

        Args:
            function (`Callable`, `optional`):
                The function to decorate.
            process_index (`int`, `optional`):
                The index of the process on which to run the function.

        Example:
        ```python
        # Assume we have 4 processes.
        from accelerate.state import PartialState

        state = PartialState()


        @state.on_process(process_index=2)
        def print_something():
            print(f"Printed on process {state.process_index}")


        print_something()
        "Printed on process 2"
        ```
        """
        if function is None:
            return partial(self.on_process, process_index=process_index)
        if (self.process_index == process_index) or (not self.use_distributed):
            return function
        return do_nothing

    def on_local_process(self, function: Callable[..., Any] = None, local_process_index: int = None):
        """
        Decorator that only runs the decorated function on the process with the given index on the current node.

        Args:
            function (`Callable`, *optional*):
                The function to decorate.
            local_process_index (`int`, *optional*):
                The index of the local process on which to run the function.

        Example:
        ```python
        # Assume we have 2 servers with 4 processes each.
        from accelerate import Accelerator

        accelerator = Accelerator()


        @accelerator.on_local_process(local_process_index=2)
        def print_something():
            print(f"Printed on process {accelerator.local_process_index}")


        print_something()
        # On server 1:
        "Printed on process 2"
        # On server 2:
        "Printed on process 2"
        ```
        """
        if function is None:
            return partial(self.on_local_process, local_process_index=local_process_index)
        if (self.local_process_index == local_process_index) or (not self.use_distributed):
            return function
        return do_nothing

    def print(self, *args, **kwargs):
        if self.is_local_main_process:
            print(*args, **kwargs)

    @property
    def default_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def unwrap_model(self, model, keep_fp32_wrapper: bool = True):
        return extract_model_from_parallel(model, keep_fp32_wrapper)
