# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Fused W3/W4 scalar GSQ relaxation compiled with CUDA NVRTC.

The CUDA code is optional. Callers retain the eager autograd implementation
when CUDA runtime compilation is unavailable or the geometry is unsupported.
BF16 logits retain BF16 softmax probabilities; FP32 Q/K logits retain FP32
probabilities, matching the eager candidate and gradient precision.
"""

import ctypes
import logging
import os

import torch

_CUDA_SOURCE = r'''
extern "C" {
__device__ __forceinline__ float from_bf16(unsigned short x) {
    return __uint_as_float(((unsigned int)x) << 16);
}
__device__ __forceinline__ unsigned short to_bf16(float x) {
    unsigned int bits = __float_as_uint(x);
    return (unsigned short)((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
}
__device__ __forceinline__ float bf(float x) { return from_bf16(to_bf16(x)); }
__device__ __forceinline__ float load_choice(const void* values, int offset, int fp32) {
    return fp32 ? ((const float*)values)[offset] : from_bf16(((const unsigned short*)values)[offset]);
}

__global__ void gsq_forward(
    const void* logits, const float* scales, const float* initial,
    const unsigned char* valid, const void* uniform,
    float* output, void* probability, float* expected,
    int n, int columns, int groups, int group_size, float temperature, float multiplier, int fp32)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float score[5];
    float maximum = -3.402823466e38f;
    for (int j = 0; j < 5; ++j) {
        int offset = j*n+i;
        if (!valid[offset]) { score[j] = -3.402823466e38f; continue; }
        float u = load_choice(uniform, offset, fp32);
        float noise;
        float scaled = load_choice(logits, offset, fp32) * multiplier;
        if (fp32) {
            noise = -logf(-logf(u + 1.0e-8f) + 1.0e-8f);
            score[j] = (scaled + noise) / temperature;
        } else {
            float inner = bf(-bf(logf(bf(u + 1.0e-8f))) + 1.0e-8f);
            noise = bf(-bf(logf(inner)));
            score[j] = bf(bf(bf(scaled) + noise) / temperature);
        }
        if (score[j] > maximum) maximum = score[j];
    }
    float exponent[5];
    float total = 0.0f;
    for (int j = 0; j < 5; ++j) {
        exponent[j] = score[j] == -3.402823466e38f ? 0.0f : expf(score[j] - maximum);
        total += exponent[j];
    }
    float shift_sum = 0.0f;
    for (int j = 0; j < 5; ++j) {
        float p = exponent[j] / total;
        if (fp32) ((float*)probability)[j*n+i] = p;
        else {
            unsigned short rounded = to_bf16(p);
            ((unsigned short*)probability)[j*n+i] = rounded;
            p = from_bf16(rounded);
        }
        shift_sum += p * (float)(j-2);
    }
    float value = initial[i] + shift_sum;
    expected[i] = value;
    int row = i / columns;
    int group = (i % columns) / group_size;
    float scale = scales[row*groups+group];
    output[i] = value*scale;
}

__global__ void gsq_backward(
    const float* grad_output, const void* probability,
    const float* expected, const float* scales,
    void* grad_logits, float* scale_terms,
    int n, int columns, int groups, int group_size, float temperature, float multiplier, int fp32)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float grad = grad_output[i];
    float value = expected[i];
    scale_terms[i] = grad * value;
    int row = i / columns;
    int group = (i % columns) / group_size;
    float output_grad = grad * scales[row*groups+group];
    float categorical[5];
    float dot = 0.0f;
    for (int j = 0; j < 5; ++j) {
        categorical[j] = output_grad * (float)(j-2);
        dot += categorical[j] * load_choice(probability, j*n+i, fp32);
    }
    for (int j = 0; j < 5; ++j) {
        float p = load_choice(probability, j*n+i, fp32);
        float delta = categorical[j] - dot;
        float result = (p*delta) * multiplier;
        if (fp32) ((float*)grad_logits)[j*n+i] = result / temperature;
        else ((unsigned short*)grad_logits)[j*n+i] = to_bf16(result / temperature);
    }
}

__global__ void gsq_scale_reduce(
    const float* terms, float* grad_scales, int rows, int columns, int groups, int group_size)
{
    int lane = threadIdx.x & 31;
    int index = blockIdx.x*4 + (threadIdx.x >> 5);
    if (index >= rows*groups) return;
    int row = index / groups;
    int group = index % groups;
    int start = group*group_size;
    float sum = 0.0f;
    for (int column = start + lane; column < start+group_size && column < columns;
         column += 32) {
        sum += terms[row*columns+column];
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (lane == 0) grad_scales[index] = sum;
}

__global__ void gsq_rms_norm(
    const unsigned short* input, const unsigned short* weight, unsigned short* output,
    int rows, int columns, float epsilon)
{
    int row = blockIdx.x;
    if (row >= rows) return;
    float partial = 0.0f;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        float value = from_bf16(input[row*columns+column]);
        partial += value*value;
    }
    __shared__ float sums[256];
    sums[threadIdx.x] = partial;
    __syncthreads();
    for (int offset = 128; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) sums[threadIdx.x] += sums[threadIdx.x+offset];
        __syncthreads();
    }
    float inverse = rsqrtf(sums[0]/(float)columns+epsilon);
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        float value = from_bf16(input[row*columns+column]);
        float normalized = bf(value*inverse);
        output[row*columns+column] = to_bf16(normalized*from_bf16(weight[column]));
    }
}
}
'''


_MODULES = {}
_FAILED = set()


def _check(result):
    if result[0].value != 0:
        raise RuntimeError(f'GSQ CUDA call failed: {result[0]}')
    return result[1] if len(result) == 2 else None


def _functions(device):
    if device in _FAILED:
        return None
    if device in _MODULES:
        return _MODULES[device]
    try:
        from cuda.bindings import driver, nvrtc

        torch.cuda.init()
        _check(driver.cuInit(0))
        major, minor = torch.cuda.get_device_capability(device)
        program = _check(nvrtc.nvrtcCreateProgram(_CUDA_SOURCE.encode(), b'gsq_relaxation.cu', 0, [], []))
        options = [f'--gpu-architecture=compute_{major}{minor}'.encode(), b'--fmad=false']
        compile_result = nvrtc.nvrtcCompileProgram(program, len(options), options)
        if compile_result[0].value:
            size = _check(nvrtc.nvrtcGetProgramLogSize(program))
            log = b' '*size
            _check(nvrtc.nvrtcGetProgramLog(program, log))
            raise RuntimeError(log.decode(errors='replace'))
        size = _check(nvrtc.nvrtcGetPTXSize(program))
        ptx = bytearray(size)
        _check(nvrtc.nvrtcGetPTX(program, ptx))
        _check(nvrtc.nvrtcDestroyProgram(program))
        ptx_buffer = ctypes.create_string_buffer(bytes(ptx))
        module = _check(driver.cuModuleLoadData(ctypes.addressof(ptx_buffer)))
        functions = tuple(_check(driver.cuModuleGetFunction(module, name))
                          for name in (b'gsq_forward', b'gsq_backward', b'gsq_scale_reduce', b'gsq_rms_norm'))
        _MODULES[device] = (driver, module, functions)
        return _MODULES[device]
    except Exception as error:  # noqa: BLE001 - optional CUDA setup must fall back
        logging.getLogger(__name__).warning('GSQ CUDA relaxation unavailable: %s', error)
        _FAILED.add(device)
        return None


def _launch(driver, function, blocks, tensors, scalars):
    from cuda.bindings.driver import CUstream

    values = tuple(tensor.data_ptr() for tensor in tensors) + tuple(scalars)
    types = (ctypes.c_void_p,)*len(tensors) + tuple(
        ctypes.c_float if isinstance(value, float) else ctypes.c_int for value in scalars)
    stream = CUstream(torch.cuda.current_stream(tensors[0].device).cuda_stream)
    _check(driver.cuLaunchKernel(function, blocks, 1, 1, 128, 1, 1, 0, stream,
                                 (values, types), 0))


def cuda_rms_norm(hidden, weight, epsilon):
    """Apply the fixed Llama RMSNorm during cached MLP training, if supported."""
    if (os.environ.get('GPTQMODEL_GSQ_DISABLE_CUDA_RELAXATION') == '1'
            or hidden.device.type != 'cuda' or hidden.dtype != torch.bfloat16
            or weight.device != hidden.device or weight.dtype != torch.bfloat16
            or not hidden.is_contiguous() or not weight.is_contiguous()
            or hidden.ndim != 3 or weight.shape != (hidden.shape[-1],)
            or hidden.shape[-1] > 8192):
        return None
    functions = _functions(hidden.device.index)
    if functions is None:
        return None
    driver, _, kernels = functions
    output = torch.empty_like(hidden)
    rows = hidden.shape[0]*hidden.shape[1]
    values = (hidden.data_ptr(), weight.data_ptr(), output.data_ptr(),
              rows, hidden.shape[-1], float(epsilon))
    types = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
             ctypes.c_int, ctypes.c_int, ctypes.c_float)
    from cuda.bindings.driver import CUstream

    stream = CUstream(torch.cuda.current_stream(hidden.device).cuda_stream)
    _check(driver.cuLaunchKernel(kernels[3], rows, 1, 1, 256, 1, 1, 0, stream,
                                 (values, types), 0))
    return output


class _GSQCudaRelaxation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, scales, initial, valid, uniform, group_size, temperature, multiplier):
        driver, _, functions = _functions(logits.device.index)
        rows, columns = initial.shape
        groups = scales.shape[1]
        count = rows*columns
        output = torch.empty_like(initial)
        probability = torch.empty_like(logits)
        expected = torch.empty_like(initial)
        _launch(driver, functions[0], (count+127)//128,
                (logits, scales, initial, valid, uniform, output, probability, expected),
                (count, columns, groups, group_size, temperature, multiplier,
                 int(logits.dtype == torch.float32)))
        ctx.save_for_backward(probability, expected, scales)
        ctx.geometry = (rows, columns, groups, group_size, temperature, multiplier)
        ctx.driver = driver
        ctx.functions = functions
        ctx.logit_dtype = logits.dtype
        return output

    @staticmethod
    def backward(ctx, grad_output):
        probability, expected, scales = ctx.saved_tensors
        rows, columns, groups, group_size, temperature, multiplier = ctx.geometry
        count = rows*columns
        grad_logits = torch.empty(probability.shape, dtype=ctx.logit_dtype, device=probability.device)
        scale_terms = torch.empty((rows, columns), dtype=torch.float32, device=scales.device)
        grad_scales = torch.empty_like(scales)
        _launch(ctx.driver, ctx.functions[1], (count+127)//128,
                (grad_output.contiguous(), probability, expected, scales, grad_logits, scale_terms),
                (count, columns, groups, group_size, temperature, multiplier,
                 int(ctx.logit_dtype == torch.float32)))
        _launch(ctx.driver, ctx.functions[2], (rows*groups+3)//4,
                (scale_terms, grad_scales), (rows, columns, groups, group_size))
        return grad_logits, grad_scales, None, None, None, None, None, None


def cuda_relaxed_scalar_weights(logits, scales, initial, valid, uniform, group_size,
                                temperature, multiplier):
    """Return None when the optional CUDA scalar path cannot be used."""
    if initial is None or initial.ndim != 2 or logits.ndim != 3:
        return None
    rows, columns = initial.shape
    if (os.environ.get('GPTQMODEL_GSQ_DISABLE_CUDA_RELAXATION') == '1'
            or logits.device.type != 'cuda' or logits.dtype not in (torch.bfloat16, torch.float32)
            or initial.dtype != torch.float32
            or scales.dtype != torch.float32 or uniform.dtype != logits.dtype
            or valid.dtype != torch.bool or not all(value.is_contiguous()
                                                     for value in (logits, scales, initial, valid, uniform))
            or any(value.device != logits.device for value in (scales, initial, valid, uniform))
            or isinstance(group_size, bool) or not isinstance(group_size, int) or group_size <= 0
            or logits.shape != (5, rows, columns) or valid.shape != logits.shape
            or uniform.shape != logits.shape
            or scales.shape != (rows, (columns+group_size-1)//group_size)):
        return None
    if _functions(logits.device.index) is None:
        return None
    return _GSQCudaRelaxation.apply(logits, scales, initial, valid, uniform,
                                    group_size, temperature, multiplier)
