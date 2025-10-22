#pragma once

// Ensure Machete kernels only build for Hopper (SM90/SM90A) architectures.
#if defined(__CUDA_ARCH__)
#if (__CUDA_ARCH__ < 900) || (__CUDA_ARCH__ >= 1000)
#error "Machete kernels require NVIDIA Hopper GPUs (compute capability 9.0 / SM90)."
#endif
#endif
