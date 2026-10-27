# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Fused W3/W4 scalar GSQ relaxation compiled at runtime.

Two optional backends compile the same portable kernel source:

* CUDA (``cuda-python`` + NVRTC -> PTX) on NVIDIA.
* ROCm (ctypes + HIPRTC -> HSACO/AMDGCN) on AMD. This is selected first when
  ``torch.version.hip`` is set, because ``cuda-python`` has no ROCm equivalent
  and PTX cannot be loaded by the HIP runtime.

Both are optional. Callers retain the eager autograd implementation when
runtime compilation is unavailable or the geometry is unsupported.
BF16 logits retain BF16 softmax probabilities; FP32 Q/K logits retain FP32
probabilities, matching the eager candidate and gradient precision.

``cuda_relaxed_scalar_weights`` only serves the W3/W4 geometry: it requires a
five-choice ``logits`` tensor and a non-None ``initial`` offset, while ``initial``
is None for W2. W2 therefore always uses the eager path.

Set ``GPTQMODEL_GSQ_DISABLE_CUDA_RELAXATION=1`` to force the eager path, or
``GPTQMODEL_GSQ_DISABLE_HIP_RELAXATION=1`` to skip only the HIPRTC backend.
"""

import ctypes
import logging
import os

import torch

# A single source of truth for both backends. Warp-level intrinsics are written
# against ``warpSize`` because AMD waves are 64 lanes wide while NVIDIA warps
# are 32; the synchronisation mask is 64-bit on HIP and 32-bit on CUDA.
_SOURCE = r'''
#ifdef __HIP_PLATFORM_AMD__
typedef unsigned long long gsq_mask_t;
#define GSQ_FULL_MASK ((gsq_mask_t)0xffffffffffffffffULL)
#else
typedef unsigned int gsq_mask_t;
#define GSQ_FULL_MASK ((gsq_mask_t)0xffffffffU)
#endif

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
    int lane = threadIdx.x % warpSize;
    int index = blockIdx.x*(blockDim.x/warpSize) + (threadIdx.x / warpSize);
    if (index >= rows*groups) return;
    int row = index / groups;
    int group = index % groups;
    int start = group*group_size;
    float sum = 0.0f;
    for (int column = start + lane; column < start+group_size && column < columns;
         column += warpSize) {
        sum += terms[row*columns+column];
    }
    for (int offset = warpSize/2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(GSQ_FULL_MASK, sum, offset, warpSize);
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

_KERNEL_NAMES = (b'gsq_forward', b'gsq_backward', b'gsq_scale_reduce', b'gsq_rms_norm')
_HIP_LIBRARIES = ('libhiprtc.so.7', 'libhiprtc.so')
_HIP_RUNTIME_LIBRARIES = ('libamdhip64.so.7', 'libamdhip64.so')
_HIP_LIBRARY_DIRS = (
    '/opt/rocm/lib',
    '/opt/rocm/core/lib',
    '/usr/lib/x86_64-linux-gnu',
)

_MODULES = {}
_FAILED = set()


def _find_library(names, dirs):
    """Load the first usable library, preferring names already on the loader path."""
    errors = []
    candidates = list(names) + [os.path.join(directory, name)
                                for directory in dirs for name in names]
    for candidate in candidates:
        if os.path.sep in candidate and not os.path.exists(candidate):
            continue
        try:
            return ctypes.CDLL(candidate)
        except OSError as error:
            errors.append(f'{candidate}: {error}')
    raise OSError('; '.join(errors))


class _Backend:
    """A loaded module plus the per-device launch primitive for one runtime."""

    __slots__ = ('driver', 'functions', 'hip', 'image', 'kind', 'module', 'warp_size')

    def __init__(self, kind, module, functions, hip=None, driver=None, warp_size=32, image=None):
        self.kind = kind
        self.module = module
        self.functions = functions
        self.hip = hip
        self.driver = driver
        self.warp_size = warp_size
        self.image = image

    def launch(self, index, grid, block, args):
        function = self.functions[index]
        if self.kind == 'hip':
            _hip_launch(self.hip, function, grid, block, args)
        else:
            _cuda_launch(self.driver, function, grid, block, args)


def _boxes(args):
    """Build the argument boxes; callers pass tensors plus int/float scalars."""
    boxes = []
    for value in args:
        if isinstance(value, torch.Tensor):
            boxes.append(ctypes.c_void_p(value.data_ptr()))
        elif isinstance(value, float):
            boxes.append(ctypes.c_float(value))
        else:
            boxes.append(ctypes.c_int(value))
    return boxes


def _hip_launch(hip, function, grid, block, args):
    boxes = _boxes(args)
    pointers = (ctypes.c_void_p * len(boxes))(
        *[ctypes.cast(ctypes.pointer(box), ctypes.c_void_p) for box in boxes])
    device = args[0].device if isinstance(args[0], torch.Tensor) else None
    stream = ctypes.c_void_p(torch.cuda.current_stream(device).cuda_stream)
    result = hip.hipModuleLaunchKernel(function, grid, 1, 1, block, 1, 1, 0,
                                       stream, pointers, None)
    if result != 0:
        raise RuntimeError(f'GSQ HIP kernel launch failed: {result}')
    # Keep the argument storage alive until the launch has been enqueued.
    del boxes


def _cuda_launch(driver, function, grid, block, args):
    from cuda.bindings.driver import CUstream

    values = tuple(value.data_ptr() if isinstance(value, torch.Tensor) else value
                   for value in args)
    types = tuple(ctypes.c_void_p if isinstance(value, torch.Tensor)
                  else ctypes.c_float if isinstance(value, float) else ctypes.c_int
                  for value in args)
    device = args[0].device if isinstance(args[0], torch.Tensor) else None
    stream = CUstream(torch.cuda.current_stream(device).cuda_stream)
    result = driver.cuLaunchKernel(function, grid, 1, 1, block, 1, 1, 0, stream,
                                   (values, types), 0)
    if result[0].value != 0:
        raise RuntimeError(f'GSQ CUDA kernel launch failed: {result[0]}')


def _hip_backend(device):
    """Compile the kernel with HIPRTC and load the HSACO image with the HIP runtime."""
    hiprtc = _find_library(_HIP_LIBRARIES, _HIP_LIBRARY_DIRS)
    hip = _find_library(_HIP_RUNTIME_LIBRARIES, _HIP_LIBRARY_DIRS)

    hiprtc.hiprtcCreateProgram.restype = ctypes.c_int
    program = ctypes.c_void_p()
    source = _SOURCE.encode()
    name = b'gsq_relaxation.hip'
    result = hiprtc.hiprtcCreateProgram(ctypes.byref(program), source, name, 0, None, None)
    if result != 0:
        raise RuntimeError(f'hiprtcCreateProgram failed: {result}')
    try:
        arch = torch.cuda.get_device_properties(device).gcnArchName.split(':')[0]
        options = [f'--gpu-architecture={arch}'.encode(), b'-ffp-contract=off']
        array = (ctypes.c_char_p * len(options))(*options)
        result = hiprtc.hiprtcCompileProgram(program, len(options), array)
        if result != 0:
            size = ctypes.c_size_t()
            hiprtc.hiprtcGetProgramLogSize(program, ctypes.byref(size))
            log = ctypes.create_string_buffer(max(size.value, 1))
            hiprtc.hiprtcGetProgramLog(program, log)
            raise RuntimeError(log.raw.decode(errors='replace').strip())
        size = ctypes.c_size_t()
        if hiprtc.hiprtcGetCodeSize(program, ctypes.byref(size)) != 0 or not size.value:
            raise RuntimeError('hiprtcGetCodeSize failed')
        image = ctypes.create_string_buffer(size.value)
        if hiprtc.hiprtcGetCode(program, image) != 0:
            raise RuntimeError('hiprtcGetCode failed')
    finally:
        hiprtc.hiprtcDestroyProgram(ctypes.byref(program))

    module = ctypes.c_void_p()
    result = hip.hipModuleLoadData(ctypes.byref(module), image)
    if result != 0:
        raise RuntimeError(f'hipModuleLoadData failed: {result}')
    functions = []
    for kernel in _KERNEL_NAMES:
        function = ctypes.c_void_p()
        result = hip.hipModuleGetFunction(ctypes.byref(function), module, kernel)
        if result != 0:
            raise RuntimeError(f'hipModuleGetFunction({kernel!r}) failed: {result}')
        functions.append(function)
    warp_size = int(torch.cuda.get_device_properties(device).warp_size)
    return _Backend('hip', module, tuple(functions), hip=hip,
                    warp_size=warp_size, image=image)


def _cuda_backend(device):
    from cuda.bindings import driver, nvrtc

    def check(result):
        if result[0].value != 0:
            raise RuntimeError(f'GSQ CUDA call failed: {result[0]}')
        return result[1] if len(result) == 2 else None

    check(driver.cuInit(0))
    major, minor = torch.cuda.get_device_capability(device)
    program = check(nvrtc.nvrtcCreateProgram(_SOURCE.encode(), b'gsq_relaxation.cu', 0, [], []))
    options = [f'--gpu-architecture=compute_{major}{minor}'.encode(), b'--fmad=false']
    result = nvrtc.nvrtcCompileProgram(program, len(options), options)
    if result[0].value:
        size = check(nvrtc.nvrtcGetProgramLogSize(program))
        log = b' '*size
        check(nvrtc.nvrtcGetProgramLog(program, log))
        raise RuntimeError(log.decode(errors='replace'))
    size = check(nvrtc.nvrtcGetPTXSize(program))
    ptx = bytearray(size)
    check(nvrtc.nvrtcGetPTX(program, ptx))
    check(nvrtc.nvrtcDestroyProgram(program))
    buffer = ctypes.create_string_buffer(bytes(ptx))
    module = check(driver.cuModuleLoadData(ctypes.addressof(buffer)))
    functions = tuple(check(driver.cuModuleGetFunction(module, kernel))
                      for kernel in _KERNEL_NAMES)
    return _Backend('cuda', module, functions, driver=driver,
                    warp_size=int(torch.cuda.get_device_properties(device).warp_size),
                    image=buffer)


def _disabled():
    """Escape hatches are checked before any cached module so they always apply."""
    if os.environ.get('GPTQMODEL_GSQ_DISABLE_CUDA_RELAXATION') == '1':
        return True
    return (torch.version.hip is not None
            and os.environ.get('GPTQMODEL_GSQ_DISABLE_HIP_RELAXATION') == '1')


def _functions(device):
    if _disabled() or device in _FAILED:
        return None
    if device in _MODULES:
        return _MODULES[device]
    try:
        torch.cuda.init()
        backend = _hip_backend(device) if torch.version.hip is not None else _cuda_backend(device)
        _MODULES[device] = backend
        return backend
    except Exception as error:  # noqa: BLE001 - optional backend must fall back
        logging.getLogger(__name__).warning('GSQ fused relaxation unavailable: %s', error)
        _FAILED.add(device)
        return None


def cuda_rms_norm(hidden, weight, epsilon):
    """Apply the fixed Llama RMSNorm during cached MLP training, if supported."""
    if (_disabled()
            or hidden.device.type != 'cuda' or hidden.dtype != torch.bfloat16
            or weight.device != hidden.device or weight.dtype != torch.bfloat16
            or not hidden.is_contiguous() or not weight.is_contiguous()
            or hidden.ndim != 3 or weight.shape != (hidden.shape[-1],)
            or hidden.shape[-1] > 8192):
        return None
    backend = _functions(hidden.device.index)
    if backend is None:
        return None
    output = torch.empty_like(hidden)
    rows = hidden.shape[0]*hidden.shape[1]
    backend.launch(3, rows, 256,
                   (hidden, weight, output, rows, hidden.shape[-1], float(epsilon)))
    return output


class _GSQCudaRelaxation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, scales, initial, valid, uniform, group_size, temperature, multiplier):
        backend = _functions(logits.device.index)
        if backend is None:
            raise RuntimeError('GSQ fused relaxation backend disappeared')
        rows, columns = initial.shape
        groups = scales.shape[1]
        count = rows*columns
        output = torch.empty_like(initial)
        probability = torch.empty_like(logits)
        expected = torch.empty_like(initial)
        fp32 = int(logits.dtype == torch.float32)
        backend.launch(0, (count+127)//128, 128,
                       (logits, scales, initial, valid, uniform, output, probability, expected,
                        count, columns, groups, group_size, temperature, multiplier, fp32))
        ctx.save_for_backward(probability, expected, scales)
        ctx.geometry = (rows, columns, groups, group_size, temperature, multiplier)
        ctx.backend = backend
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
        backend = ctx.backend
        fp32 = int(ctx.logit_dtype == torch.float32)
        backend.launch(1, (count+127)//128, 128,
                       (grad_output.contiguous(), probability, expected, scales,
                        grad_logits, scale_terms,
                        count, columns, groups, group_size, temperature, multiplier, fp32))
        per_block = max(128 // backend.warp_size, 1)
        backend.launch(2, (rows*groups+per_block-1)//per_block, 128,
                       (scale_terms, grad_scales, rows, columns, groups, group_size))
        return grad_logits, grad_scales, None, None, None, None, None, None


def cuda_relaxed_scalar_weights(logits, scales, initial, valid, uniform, group_size,
                                temperature, multiplier):
    """Return None when the optional fused scalar path cannot be used."""
    if initial is None or initial.ndim != 2 or logits.ndim != 3:
        return None
    rows, columns = initial.shape
    if (_disabled()
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
