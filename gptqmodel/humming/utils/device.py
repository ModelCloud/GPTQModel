# Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
# SPDX-License-Identifier: Apache-2.0
#
# This file is vendored from https://github.com/inclusionAI/humming and used under
# the terms of the Apache License, Version 2.0.
# See gptqmodel/humming/LICENSE for the full license text.

import pynvml
import torch


def get_device_name(gpu_index=0):
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        return pynvml.nvmlDeviceGetName(handle)
    finally:
        pynvml.nvmlShutdown()


def calculate_gpu_bandwidth(gpu_index=0):
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
        if (major, minor) == (12, 1):
            # NVIDIA GB10 (DGX Spark, sm_121): unified LPDDR5X memory with no
            # discrete memory-clock domain, so NVML_CLOCK_MEM raises
            # NVML_ERROR_NOT_SUPPORTED (and the reported bus width is 0).
            # Use the platform's known memory bandwidth instead.
            return 273.0
        gpu_name = pynvml.nvmlDeviceGetName(handle)
        try:
            bus_width = pynvml.nvmlDeviceGetMemoryBusWidth(handle)
        except pynvml.NVMLError_FunctionNotFound:
            # nvidia driver 470 + cuda-compat supports cuda 12
            # but doesn't support nvmlDeviceGetMemoryBusWidth.
            # so we hardcode bus width for some old devices.
            if "A100" in gpu_name or "A800" in gpu_name:
                bus_width = 5120
            elif "A10" in gpu_name:
                bus_width = 384
            elif "T4" in gpu_name:
                bus_width = 256
            else:
                raise
        mem_clock_mhz = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_MEM)
        return (mem_clock_mhz * 2 * bus_width) / 8 / 1000
    finally:
        pynvml.nvmlShutdown()


def estimate_tensorcore_max_tops(gpu_index=0):
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
        sm_version = major * 10 + minor
        max_clock_mhz = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_SM)
        sm_count = get_device_num_sms(gpu_index)

        ops_map = {
            75: 1024,
            80: 2048,
            86: 1024,
            87: 2048,
            89: 1024,
            90: 4096,
            100: 8192,
            103: 8192,
            120: 1024,
            121: 1024,
        }
        ops_per_clock = ops_map[sm_version]

        # 1. This function returns the dense FP16 Tensor Core performance (FP16 accumulator).
        #    Note that on certain architectures (such as SM75/SM86/SM89),
        #    the performance of the FP32 accumulator is only half that of the FP16 accumulator.
        # 2. Due to power limiting (power walls), most GPUs cannot reach the max clock speed
        #    that read by nvml. The actual achievable peak frequency must be determined
        #    through real-world benchmarking. We only estimate the value.
        factor = 0.9 if sm_version != 80 else 1.0
        return (sm_count * ops_per_clock * max_clock_mhz) / 1e6 * factor
    finally:
        pynvml.nvmlShutdown()


def estimate_compute_bound_threshold(weight_nbytes, shape_n, shape_k, dtype, use_f16_accum):
    # total_memory_size = weight_nbytes + shape_k * shape_m * dtype.num_bits / 8
    # total_compute_ops = shape_n * shape_k * shape_m * 2
    # given (total_memory_size / max_bandwidth) = (total_compute_ops / max_tops), solve shape_m
    max_bandwidth = calculate_gpu_bandwidth()
    max_tops = estimate_tensorcore_max_tops()
    num_bits = 16
    if dtype in ["float8e4m3", "float8e5m2", "int8"]:
        max_tops = max_tops * 2
        num_bits = 8
    elif dtype in ["int4", "float4e2m1"]:
        max_tops = max_tops * 4
        num_bits = 4
    sm_version_tuple = torch.cuda.get_device_capability()
    if sm_version_tuple in [(7, 5), (8, 6), (8, 9)] and "float" in dtype and not use_f16_accum:
        max_tops = max_tops / 2

    left_bias = weight_nbytes / max_bandwidth
    left_factor = shape_k * num_bits / 8 / max_bandwidth
    right_factor = shape_n * shape_k * 2 / max_tops

    return left_bias / (right_factor - left_factor) * 1e3


def get_device_num_sms(gpu_index: int | None = None):
    if gpu_index is None:
        default_device = torch.get_default_device()
        if default_device.type == "cuda":
            gpu_index = default_device.index
        else:
            gpu_index = 0

    dev_props = torch.cuda.get_device_properties(gpu_index)

    return dev_props.multi_processor_count
