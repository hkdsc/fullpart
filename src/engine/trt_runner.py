import abc
import ctypes
import os
import sys

# from .parser import Parser, init_tensorrt_pybind
# from . import _kwainn_tensorrt_pybind
from abc import abstractmethod

from cuda import cuda, cudart

PY2 = sys.version_info[0] == 2

# if os.name == "nt":
#     kernel32 = ctypes.WinDLL("kernel32.dll", use_last_error=True)
#     with_load_library_flags = hasattr(kernel32, "AddDllDirectory")
#     prev_error_mode = kernel32.SetErrorMode(0x0001)
#     dll_path = os.path.dirname(__file__)
#     if sys.version_info >= (3, 8):
#         os.add_dll_directory(dll_path)
#     elif with_load_library_flags:
#         kernel32.AddDllDirectory.restype = ctypes.c_void_p
#         res = kernel32.AddDllDirectory(dll_path)
#         if res is None:
#             err = ctypes.WinError(ctypes.get_last_error())
#             err.strerror += " Error adding " + dll_path + " to the DLL directories."
#             raise err
#     for file_name in os.listdir(dll_path):
#         if (file_name.startswith("cu") or file_name.startswith("nv")) and file_name.endswith(".dll"):
#             try:
#                 ctypes.cdll.LoadLibrary(file_name)
#             except OSError:
#                 kernel32.LoadLibraryExW.restype = ctypes.c_void_p
#                 dll = os.path.join(dll_path, file_name)
#                 res = kernel32.LoadLibraryExW(dll, None, 0x00001100)
#                 if res is None:
#                     err = ctypes.WinError(ctypes.get_last_error())
#                     err.strerror += " Error loading " + dll + " or one of its dependencies."
#                     raise err
#     kernel32.SetErrorMode(prev_error_mode)


class DeviceMemoryOptions:
    def __init__(self):
        self.cuda_device_id = None

    __slots__ = [
        "cuda_device_id",
    ]


class RunnerOptions:
    def __init__(self):
        self.cuda_device_id = None
        self.severity = 3  # ERROR=1 WARNING=2 INFO=3 VERBOSE=4
        self.use_legacy_parser = False
        self.num_execution_contexts_max = 1
        self.use_shared_device_memory = False

    __slots__ = [
        "cuda_device_id",
        "severity",
        "use_legacy_parser",
        "num_execution_contexts_max",
        "use_shared_device_memory",
    ]


class CompileOptions:
    def __init__(self):
        self.enable_fp16 = True
        self.enable_int8 = False
        self.max_workspace_size = 1 << 28  # 256MiB
        self.half_io = False
        self.dynamic_shape_dict = dict()
        self.optimization_profiles = list()
        self.refittable = False
        self.timing_cache_file_path = None
        self.timing_cache_udpate = True
        self.verbose_profiling = False
        self.enable_sparse_weights = False
        self.enable_tactic_sources = True
        self.enable_preview_feature_faster_dynamic_shapes = True

    __slots__ = [
        "enable_fp16",
        "enable_int8",
        "max_workspace_size",
        "half_io",
        "dynamic_shape_dict",
        "optimization_profiles",
        "refittable",
        "timing_cache_file_path",
        "timing_cache_udpate",
        "verbose_profiling",
        "enable_sparse_weights",
        "enable_tactic_sources",
        "enable_preview_feature_faster_dynamic_shapes",
    ]


class DeviceMemory:
    def __init__(self, device_memory_options):
        self.c_device_memory = _kwainn_tensorrt_pybind.DeviceMemory(device_memory_options)

    def size(self):
        return self.c_device_memory.size()

    def resize(self, new_size):
        return self.c_device_memory.resize(new_size)


class Runner(object if PY2 else abc.ABC):
    if PY2:
        __metaclass__ = abc.ABCMeta

    def __init__(self, runner_options):
        super(Runner, self).__init__()

    @staticmethod
    @abstractmethod
    def get_tensorrt_version():
        pass

    def load(self, file_path):
        with open(file_path, "rb") as f:
            buffer = f.read()
        return self.load_buffer(buffer, os.path.splitext(file_path)[1])

    @abstractmethod
    def load_buffer(self, buffer, filename_ext):
        pass

    @abstractmethod
    def mark_output(self, tensor_name):
        pass

    @abstractmethod
    def clear_outputs(self):
        pass

    @abstractmethod
    def compile(self, compile_options):
        pass

    def load_engine(self, engine_file_path, inputs_order=None, outputs_order=None):
        with open(engine_file_path, "rb") as f:
            buffer = f.read()
        if not buffer:
            if not buffer:
                raise RuntimeError("empty engine file")
        return self.load_engine_buffer(buffer, inputs_order, outputs_order)

    @abstractmethod
    def load_engine_buffer(self, engine_buffer, inputs_order=None, outputs_order=None):
        pass

    @abstractmethod
    def serialize(self):
        pass

    @abstractmethod
    def get_engine_information(self):
        pass

    @abstractmethod
    def _infer_dlpack(self, inputs):
        pass

    @abstractmethod
    def get_device_memory_size(self):
        pass

    @abstractmethod
    def set_shared_device_memory(self, device_memory):
        pass

    def infer_dlpack(self, *inputs):
        if len(inputs) == 1 and isinstance(inputs[0], (tuple, list)):
            inputs = inputs[0]
        outputs = self._infer_dlpack(inputs)
        if len(outputs) == 1:
            return outputs[0]
        return outputs

    def _infer_torch(self, inputs):
        import torch.utils.dlpack

        inputs = [torch.utils.dlpack.to_dlpack(t.contiguous()) for t in inputs]
        outputs = self._infer_dlpack(inputs)
        outputs = [torch.utils.dlpack.from_dlpack(t) for t in outputs]
        return outputs

    def infer_torch(self, *inputs):
        if len(inputs) == 1 and isinstance(inputs[0], (tuple, list)):
            inputs = inputs[0]
        outputs = self._infer_torch(inputs)
        if len(outputs) == 1:
            return outputs[0]
        return outputs

    def mark_outputs(self, tensors_names):
        for tensor_name in tensors_names:
            self.mark_output(tensor_name)


def check(condition, msg="check failed"):
    success = bool(condition)
    if not success:
        assert success, msg
        raise RuntimeError(msg)


def get_cuda_runtime_version():
    # return _kwainn_tensorrt_pybind.cudart_helper.get_cuda_runtime_version()
    return cudaRuntimeGetVersion()


def get_cuda_device():
    # return _kwainn_tensorrt_pybind.cudart_helper.get_cuda_device()
    return cudart.cudaGetDevice()[1]


def set_cuda_device(device_id):
    # return _kwainn_tensorrt_pybind.cudart_helper.set_cuda_device(device_id)
    return cudart.cudaSetDevice(device_id)


class CudaDeviceGuard:
    def __init__(self, cuda_device_id):
        self.device_id = cuda_device_id

    def __enter__(self):
        self.old_device_id = get_cuda_device()
        set_cuda_device(self.device_id)

    def __exit__(self, *exc):
        set_cuda_device(self.old_device_id)


class PyRunner(Runner):
    def __init__(self, runner_options):
        super(PyRunner, self).__init__(runner_options)
        import tensorrt

        self.trt = tensorrt
        self.logger = self.trt.Logger(tensorrt.tensorrt.ILogger.Severity(runner_options.severity))
        if runner_options.cuda_device_id is None:
            cuda_device_id = get_cuda_device()
        else:
            cuda_device_id = runner_options.cuda_device_id
        self.cuda_device_guard = CudaDeviceGuard(cuda_device_id)
        self.use_legacy_parser = runner_options.use_legacy_parser
        if runner_options.num_execution_contexts_max != 1:
            raise NotImplementedError("not implemented multiple execution contexts for PyRunner")
        if runner_options.use_shared_device_memory:
            raise NotImplementedError("not implemented shared device memory for PyRunner")

    @staticmethod
    def get_tensorrt_version():
        import tensorrt

        NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH, _ = tensorrt.__version__.split(".")
        return (int(NV_TENSORRT_MAJOR) * 1000) + (int(NV_TENSORRT_MINOR) * 100) + int(NV_TENSORRT_PATCH)

    def load_buffer(self, buffer, filename_ext):
        with self.cuda_device_guard:
            self.builder = self.trt.Builder(self.logger)
        EXPLICIT_BATCH = 1 << (int)(self.trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        self.network = self.builder.create_network(EXPLICIT_BATCH)
        if filename_ext == ".onnx":
            self.parser = self.trt.OnnxParser(self.network, self.logger)
            if not self.parser.parse(buffer):
                error = self.parser.get_error(0)
                msg = "While parsing node number %i:\n" % error.node()
                msg += "%s:%i In function %s:\n[%i] %s" % (error.file(), error.line(), error.func(), error.code(), error.desc())
                raise RuntimeError(msg)
        # else:
        #     init_tensorrt_pybind()
        #     if self.use_legacy_parser:
        #         raise NotImplementedError("not implemented legacy parser for PyRunner")
        #     self.parser = Parser(self.network, self.logger)
        #     if not self.parser.parse(buffer):
        #         raise RuntimeError("Failed to parse the KwaiNN model file.")

    def mark_output(self, tensor_name):
        return self.parser.mark_output(tensor_name)

    def clear_outputs(self):
        for _ in range(self.network.num_outputs):
            self.network.unmark_output(self.network.get_output(0))

    def compile(self, compile_options):
        config = self.builder.create_builder_config()
        if compile_options.enable_fp16:
            config.set_flag(self.trt.BuilderFlag.FP16)
        if compile_options.enable_int8:
            config.set_flag(self.trt.BuilderFlag.INT8)
        config.max_workspace_size = compile_options.max_workspace_size

        if compile_options.half_io:
            for input_id in range(self.network.num_inputs):
                tensor = self.network.get_input(input_id)
                if tensor.dtype == self.trt.DataType.FLOAT:
                    tensor.dtype = self.trt.DataType.HALF
            for output_id in range(self.network.num_outputs):
                tensor = self.network.get_output(output_id)
                if tensor.dtype == self.trt.DataType.FLOAT:
                    tensor.dtype = self.trt.DataType.HALF

        if compile_options.dynamic_shape_dict:
            profile = self.builder.create_optimization_profile()
            for input_id, dynamic_shape in compile_options.dynamic_shape_dict.items():
                input = self.network.get_input(input_id)
                input.shape = [a if a == b and b == c else -1 for a, b, c in zip(*dynamic_shape)]
                profile.set_shape(input.name, *dynamic_shape)
            config.add_optimization_profile(profile)

        if self.get_tensorrt_version() >= 8000 and self.get_tensorrt_version() < 8100 and get_cuda_runtime_version() < 11000:
            check(config.set_tactic_sources(1 << int(self.trt.TacticSource.CUBLAS)))
        if not compile_options.enable_tactic_sources:
            check(config.set_tactic_sources(0))
        if self.get_tensorrt_version() >= 8500 and compile_options.enable_preview_feature_faster_dynamic_shapes:
            config.set_preview_feature(self.trt.PreviewFeature.FASTER_DYNAMIC_SHAPES_0805, True)

        if compile_options.timing_cache_file_path:
            if os.path.isfile(compile_options.timing_cache_file_path):
                with open(compile_options.timing_cache_file_path, "rb") as f:
                    timing_cache = config.create_timing_cache(f.read())
            else:
                timing_cache = config.create_timing_cache(b"")
            check(config.set_timing_cache(timing_cache, ignore_mismatch=False))

        if compile_options.verbose_profiling:
            config.profiling_verbosity = self.trt.ProfilingVerbosity.DETAILED

        with self.cuda_device_guard:
            if self.get_tensorrt_version() >= 8000:
                self.plan = self.builder.build_serialized_network(self.network, config)
                if not self.plan:
                    raise RuntimeError("build_serialized_network failed")
                runtime = self.trt.Runtime(self.logger)
                self.engine = runtime.deserialize_cuda_engine(self.plan)
                if not self.engine:
                    raise RuntimeError("deserialize_cuda_engine failed")
            else:
                self.engine = self.builder.build_engine(self.network, config)
                if not self.engine:
                    raise RuntimeError("build_engine failed")
                self.plan = self.engine.serialize()
                if not self.plan:
                    raise RuntimeError("engine serialize failed")
            self.execution_context = self.engine.create_execution_context()
            if not self.execution_context:
                raise RuntimeError("create_execution_context failed")

        if compile_options.timing_cache_udpate and compile_options.timing_cache_file_path:
            timing_cache_host_memory = timing_cache.serialize()
            with open(compile_options.timing_cache_file_path, "wb") as f:
                f.write(timing_cache_host_memory)

        inputs_order = [self.network.get_input(i).name for i in range(self.network.num_inputs)]
        outputs_order = [self.network.get_output(i).name for i in range(self.network.num_outputs)]
        self.update_binding_order(inputs_order, outputs_order)

    def load_engine_buffer(self, engine_buffer, inputs_order=None, outputs_order=None):
        self.plan = engine_buffer
        with self.cuda_device_guard:
            runtime = self.trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(self.plan)
            if not self.engine:
                raise RuntimeError("deserialize_cuda_engine failed")
            self.execution_context = self.engine.create_execution_context()
            if not self.execution_context:
                raise RuntimeError("create_execution_context failed")
        if inputs_order is None:
            inputs_order = list()
            for binding_name in self.engine:
                if self.engine.binding_is_input(binding_name):
                    inputs_order.append(binding_name)
            check(len(inputs_order) <= 1, "input order must be explicitly provided if #input > 1")
        if outputs_order is None:
            outputs_order = list()
            for binding_name in self.engine:
                if not self.engine.binding_is_input(binding_name):
                    outputs_order.append(binding_name)
            check(len(outputs_order) <= 1, "output order must be explicitly provided if #output > 1")
        self.update_binding_order(inputs_order, outputs_order)

    def serialize(self):
        return bytes(self.plan)

    def get_engine_information(self):
        inspector = self.engine.create_engine_inspector()
        inspector.execution_context = self.execution_context
        return inspector.get_engine_information(self.trt.LayerInformationFormat.JSON)

    def _infer_dlpack(self, inputs):
        # TODO: no torch
        import torch
        import torch.utils.dlpack

        runner_device = torch.device("cuda", self.cuda_device_guard.device_id)
        inputs = [torch.utils.dlpack.from_dlpack(t) for t in inputs]
        check(all(t.device == runner_device and t.is_contiguous() for t in inputs))
        for input_id, input in enumerate(inputs):
            self.execution_context.set_binding_shape(input_id, input.shape)
        num_inputs = sum(t[0] for t in self.binding_order)  # t[0]: is_input
        check(len(inputs) == num_inputs, "inputs number mismatch")
        outputs = [None for t in self.binding_order if not t[0]]  # t[0]: is_input
        bindings = list()
        map_dtype_trt_to_torch = {
            self.trt.DataType.FLOAT: torch.float32,
            self.trt.DataType.HALF: torch.half,
            self.trt.DataType.INT32: torch.int32,
            self.trt.DataType.BF16: torch.bfloat16,
        }
        for is_input, binding_id, io_id in self.binding_order:
            dtype = self.engine.get_binding_dtype(binding_id)
            shape = self.execution_context.get_binding_shape(binding_id)
            torch_dtype = map_dtype_trt_to_torch[dtype]
            if is_input:
                check(inputs[binding_id].dtype == torch_dtype)
                bindings.append(inputs[io_id].data_ptr())
            else:
                # TODO: no malloc
                tensor = torch.empty(*shape, dtype=torch_dtype, device=runner_device)
                outputs[io_id] = tensor
                bindings.append(tensor.data_ptr())
        with self.cuda_device_guard:
            check(self.execution_context.execute_v2(bindings=bindings))
        outputs = [torch.utils.dlpack.to_dlpack(t) for t in outputs]
        return outputs

    def get_device_memory_size(self):
        return self.engine.device_memory_size

    def set_shared_device_memory(self, device_memory):
        raise NotImplementedError("not implemented shared device memory for PyRunner")

    def update_binding_order(self, inputs_order, outputs_order):
        self.binding_order = list()
        satisfied_outputs = set()
        for binding_id, binding_name in enumerate(self.engine):
            if self.engine.binding_is_input(binding_id):
                self.binding_order.append((True, binding_id, inputs_order.index(binding_name)))
            else:
                # TODO: allow missing outputs
                self.binding_order.append((False, binding_id, outputs_order.index(binding_name)))
                satisfied_outputs.add(outputs_order.index(binding_name))
        check(len(satisfied_outputs) == len(outputs_order), "there are unsatisfied outputs in outputs_order")


# class CRunner(Runner):
#     def __init__(self, runner_options):
#         super(CRunner, self).__init__(runner_options)
#         self.c_runner = _kwainn_tensorrt_pybind.Runner(runner_options)

#     @staticmethod
#     def get_tensorrt_version():
#         return _kwainn_tensorrt_pybind.nvinfer1_getInferLibVersion()

#     def load_buffer(self, buffer, filename_ext):
#         return self.c_runner.load_buffer(buffer, filename_ext)

#     def mark_output(self, tensor_name):
#         return self.c_runner.mark_output(tensor_name)

#     def clear_outputs(self):
#         return self.c_runner.clear_outputs()

#     def compile(self, compile_options):
#         return self.c_runner.compile(compile_options)

#     def load_engine_buffer(self, engine_buffer, inputs_order=None, outputs_order=None):
#         return self.c_runner.load_engine_buffer(engine_buffer, inputs_order or [], outputs_order or [])

#     def serialize(self):
#         return self.c_runner.serialize()

#     def get_engine_information(self):
#         return self.c_runner.get_engine_information()

#     def _infer_dlpack(self, inputs):
#         return self.c_runner._infer_dlpack(inputs)

#     def _infer_torch(self, inputs):
#         import torch
#         inputs = [t.contiguous() for t in inputs]
#         device = torch.device("cuda", self.get_cuda_device_id())
#         input_shapes = [t.shape for t in inputs]
#         output_shapes = self.c_runner._reshape(input_shapes)
#         input_dtypes, output_dtypes = self.c_runner.get_io_dtype()
#         input_dtypes = [self.convert_dtype(d) for d in input_dtypes]
#         output_dtypes = [self.convert_dtype(d) for d in output_dtypes]
#         for i, input in enumerate(inputs):
#             if input.device != device:
#                 raise ValueError("input {:d} device mismatch, expecting {:s} got {:s}".format(i, str(device), str(input.device)))
#             if input.dtype != input_dtypes[i]:
#                 raise ValueError("input {:d} dtype mismatch, expecting {:s} got {:s}".format(i, str(input_dtypes[i]), str(input.dtype)))
#         outputs = [torch.empty(s, dtype=d, device=device) for s, d in zip(output_shapes, output_dtypes)]
#         self.c_runner._enqueue([t.data_ptr() for t in outputs], [t.data_ptr() for t in inputs], torch.cuda.current_stream(device).cuda_stream)
#         return outputs

#     def _enqueue(self, outputs, inputs, stream):
#         return self.c_runner._enqueue(outputs, inputs, stream)

#     def get_cuda_device_id(self):
#         return self.c_runner.get_cuda_device_id()

#     def get_device_memory_size(self):
#         return self.c_runner.get_device_memory_size()

#     def set_shared_device_memory(self, device_memory):
#         return self.c_runner.set_shared_device_memory(device_memory.c_device_memory)

#     dtype_map = {
#         (0, 8, 1): "int8",
#         (0, 16, 1): "int16",
#         (0, 32, 1): "int32",
#         (0, 64, 1): "int64",
#         (1, 8, 1): "uint8",
#         (2, 16, 1): "float16",
#         (2, 32, 1): "float32",
#         (2, 64, 1): "float64",
#         (4, 16, 1): "bfloat16",
#         (5, 32, 1): "complex32",
#         (5, 64, 1): "complex64",
#         (5, 128, 1): "complex128",
#     }

#     @classmethod
#     def convert_dtype(cls, dlpack_dtype):
#         import torch
#         dtype_name = cls.dtype_map[dlpack_dtype]
#         return getattr(torch, dtype_name)
