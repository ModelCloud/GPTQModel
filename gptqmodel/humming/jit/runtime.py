# Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
# SPDX-License-Identifier: Apache-2.0
#
# This file is vendored from https://github.com/inclusionAI/humming and used under
# the terms of the Apache License, Version 2.0.
# See gptqmodel/humming/LICENSE for the full license text.

import dataclasses
import os
import threading
from typing import Any, ClassVar

import cuda.bindings.driver as cbd
import torch

from .. import dtypes
from .compiler import NVCCCompiler, NVRTCCompiler


@dataclasses.dataclass(kw_only=True)
class KernelRuntime:
    disable_fast_math: ClassVar[bool] = False
    _instances: ClassVar[dict[tuple[str, tuple[Any, ...]], "KernelRuntime"]] = {}

    def __new__(cls, *args, **kwargs):
        def get_value(value):
            if isinstance(value, dtypes.DataType):
                return str(value)
            if isinstance(value, list):
                value = tuple(value)
            return value

        args_items = tuple(get_value(x) for x in args)
        kwargs_items = tuple((key, get_value(kwargs[key])) for key in sorted(kwargs.keys()))
        signature = (cls.__name__, args_items + kwargs_items)

        if signature not in cls._instances or not cls._instances[signature].inited:
            instance = super().__new__(cls)
            cls._instances[signature] = instance
            instance.inited = False
            instance.cubin_loaded = False
        return cls._instances[signature]

    def __post_init__(self):
        if self.inited:
            return
        self.init_sm_version()
        self.init_kernel()
        self.inited = True
        if not hasattr(self, "_vars"):
            self._vars = vars(self)
        else:
            for key, value in self._vars.items():
                setattr(self, key, value)

    def init_kernel(self):
        raise NotImplementedError

    def init_sm_version(self):
        device_props = torch.cuda.get_device_properties()
        sm_version = device_props.major * 10 + device_props.minor
        self.sm_version = sm_version
        self.sm_version_str = str(sm_version)
        if self.sm_version >= 90:
            self.sm_version_str += "a"

    @staticmethod
    def _get_compiler():
        compiler = os.environ.get("HUMMING_COMPILER", "").lower()
        if compiler == "nvcc":
            return NVCCCompiler
        elif compiler == "nvrtc":
            return NVRTCCompiler
        else:
            try:
                from cuda.bindings import nvrtc  # noqa

                return NVRTCCompiler
            except Exception:
                return NVCCCompiler

    @staticmethod
    def _ensure_cuda_context():
        torch.cuda.set_device(torch.cuda.current_device())

    def prepare(self):
        self._ensure_cuda_context()
        compiler_cls = self._get_compiler()
        kernel_expr = getattr(self, "kernel_expr", None)
        kernel_filename = compiler_cls.compile(
            self.code,
            sm_version=self.sm_version_str,
            kernel_expr=kernel_expr,
            disable_fast_math=self.disable_fast_math,
            postprocess_cubin=self.postprocess_cubin,
        )
        self.kernel_filename = kernel_filename
        if threading.current_thread() is threading.main_thread():
            self.load_cubin()

    def postprocess_cubin(self, cubin_path: str):
        pass

    def load_cubin(self):
        if self.cubin_loaded:
            return None
        kernel_filename = self.kernel_filename
        result, lib = cbd.cuLibraryLoadFromFile(kernel_filename.encode(), [], [], 0, [], [], 0)
        assert result == 0, repr(result)
        result, num_kernels = cbd.cuLibraryGetKernelCount(lib)
        assert result == 0, repr(result)
        result, kernels = cbd.cuLibraryEnumerateKernels(num_kernels, lib)
        assert result == 0, repr(result)
        keyword = self.name.encode()
        matched = []
        for kernel in kernels:
            result, name = cbd.cuKernelGetName(kernel)
            assert result == 0, repr(result)
            if keyword in name:
                matched.append((kernel, name))
        assert len(matched) == 1, (kernel_filename, self.name, [x[1] for x in matched])
        self.kernel, kernel_name = matched[0]
        self.kernel_name = kernel_name.decode()
        result, func = cbd.cuKernelGetFunction(self.kernel)
        assert result == 0, repr(result)
        self.func = func
        self.cubin_loaded = True

    def check_context(self):
        assert threading.current_thread() is threading.main_thread()
        if not self.cubin_loaded:
            self.load_cubin()

    def __call__(self, *args, **kwargs):
        raise NotImplementedError
