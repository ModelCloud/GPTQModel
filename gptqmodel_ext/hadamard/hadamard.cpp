/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Adapted for GPT-QModel as a torch.ops JIT extension.
 ******************************************************************************/

#include <torch/extension.h>
#if TORCH_VERSION_MAJOR > 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 6)
#include <c10/core/DeviceGuard.h>
#else
#include <c10/cuda/CUDAGuard.h>
#endif
#include <c10/cuda/CUDAStream.h>
#include <vector>

#include "fast_hadamard_transform.h"

#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")

#define DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(ITYPE, NAME, ...)                    \
    if (ITYPE == at::ScalarType::Half) {                                            \
        using input_t = at::Half;                                                   \
        __VA_ARGS__();                                                              \
    } else if (ITYPE == at::ScalarType::BFloat16) {                                 \
        using input_t = at::BFloat16;                                               \
        __VA_ARGS__();                                                              \
    } else if (ITYPE == at::ScalarType::Float) {                                    \
        using input_t = float;                                                      \
        __VA_ARGS__();                                                              \
    } else {                                                                        \
        AT_ERROR(#NAME, " not implemented for input type '", toString(ITYPE), "'"); \
    }

template<typename input_t>
void fast_hadamard_transform_cuda(HadamardParamsBase &params, cudaStream_t stream);

template<typename input_t>
void fast_hadamard_transform_reverse_cuda(HadamardParamsBase &params, cudaStream_t stream);

template<typename input_t>
void fast_hadamard_transform_scaled_cuda(HadamardParamsBase &params, cudaStream_t stream);

template<typename input_t, typename output_t = input_t>
void fast_hadamard_transform_scaled_saved_cuda(HadamardParamsBase &params, cudaStream_t stream);
template<typename input_t>
void fast_hadamard_transform_sandwich_cuda(HadamardParamsBase &params, cudaStream_t stream);

template<typename input_t>
void fast_hadamard_transform_reverse_scaled_cuda(HadamardParamsBase &params, cudaStream_t stream);

template<typename input_t>
void fast_hadamard_transform_12N_cuda(HadamardParamsBase &params, cudaStream_t stream);

template<typename input_t>
void fast_hadamard_transform_20N_cuda(HadamardParamsBase &params, cudaStream_t stream);

template<typename input_t>
void fast_hadamard_transform_28N_cuda(HadamardParamsBase &params, cudaStream_t stream);

template<typename input_t>
void fast_hadamard_transform_40N_cuda(HadamardParamsBase &params, cudaStream_t stream);

void gsq_position_error_cuda(
    const at::Tensor probabilities, const at::Tensor baseline,
    const at::Tensor position_indices, const at::Tensor position_choices,
    const at::Tensor position_deltas, const at::Tensor target,
    at::Tensor error, cudaStream_t stream);

namespace gptqmodel_hadamard {

void gsq_position_error(
    const at::Tensor probabilities, const at::Tensor baseline,
    const at::Tensor position_indices, const at::Tensor position_choices,
    const at::Tensor position_deltas, const at::Tensor target,
    at::Tensor error) {
    TORCH_CHECK(probabilities.is_cuda(), "gsq_position_error expects CUDA tensors");
    TORCH_CHECK(probabilities.scalar_type() == at::ScalarType::Float,
                "GSQ probabilities must be FP32");
    for (const auto &tensor : {baseline, position_deltas, target, error}) {
        TORCH_CHECK(tensor.is_cuda() && tensor.device() == probabilities.device(),
                    "GSQ tensors must share one CUDA device");
        TORCH_CHECK(tensor.scalar_type() == at::ScalarType::BFloat16,
                    "GSQ value tensors must be BF16");
        TORCH_CHECK(tensor.is_contiguous(), "GSQ value tensors must be contiguous");
    }
    for (const auto &tensor : {position_indices, position_choices}) {
        TORCH_CHECK(tensor.is_cuda() && tensor.device() == probabilities.device(),
                    "GSQ metadata must share the CUDA device");
        TORCH_CHECK(tensor.scalar_type() == at::ScalarType::Byte && tensor.is_contiguous(),
                    "GSQ metadata must be contiguous uint8");
    }
    TORCH_CHECK(probabilities.is_contiguous(), "GSQ probabilities must be contiguous");
    TORCH_CHECK(probabilities.dim() == 2 && baseline.dim() == 2 && target.dim() == 2,
                "GSQ probabilities, baseline, and target must be matrices");
    TORCH_CHECK(position_indices.dim() == 2 && position_choices.dim() == 3 &&
                position_deltas.dim() == 3, "GSQ compact metadata has invalid rank");
    TORCH_CHECK(position_indices.size(1) == 64 && position_choices.size(2) == 3,
                "native GSQ error kernel requires P32's 64x3 compact map");
    TORCH_CHECK(position_indices.size(0) == probabilities.size(0) &&
                position_choices.sizes() == torch::IntArrayRef({probabilities.size(0), 64, 3}) &&
                position_deltas.sizes() == position_choices.sizes(),
                "GSQ compact metadata shape is invalid");
    TORCH_CHECK(target.sizes() == error.sizes() && target.size(1) % 16 == 0,
                "GSQ target/output shape is invalid");
    TORCH_CHECK(baseline.size(0) == probabilities.size(0) && baseline.size(1) == 256,
                "GSQ baseline shape is invalid");
    TORCH_CHECK(target.numel() == probabilities.size(0) * 256,
                "GSQ target does not match the tile count");

#if TORCH_VERSION_MAJOR > 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 6)
    c10::DeviceGuard device_guard(probabilities.device());
#else
    at::cuda::CUDAGuard device_guard{probabilities.device()};
#endif
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    gsq_position_error_cuda(probabilities, baseline, position_indices,
                            position_choices, position_deltas, target, error, stream);
}

void set_hadamard_params(HadamardParamsBase &params,
                         const size_t batch,
                         const size_t dim,
                         const size_t multiple,
                         const at::Tensor x,
                         const at::Tensor out,
                         double scale) {
    memset(&params, 0, sizeof(params));

    params.batch = batch;
    params.dim = dim;
    params.log_N = int(ceil(std::log2(dim / multiple)));

    params.x_ptr = x.data_ptr();
    params.out_ptr = out.data_ptr();
    params.x_batch_stride = x.stride(0);
    params.out_batch_stride = out.stride(0);

    params.scale = static_cast<float>(scale);
}


torch::Tensor fast_hadamard_transform(torch::Tensor x, double scale) {
    auto input_type = x.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::Float || input_type == at::ScalarType::Half || input_type == at::ScalarType::BFloat16);

    TORCH_CHECK(x.is_cuda(), "fast_hadamard_transform expects a CUDA tensor");

    const auto shapes_og = x.sizes();
    const int dim_og = x.size(-1);
    x = x.reshape({-1, dim_og});
    if (x.stride(-1) != 1) { x = x.contiguous(); }
    const auto sizes = x.sizes();
    const int batch_size = sizes[0];

    CHECK_SHAPE(x, batch_size, dim_og);
    TORCH_CHECK(x.stride(1) == 1);

    if (dim_og % 8 != 0) {
        x = torch::nn::functional::pad(x, torch::nn::functional::PadFuncOptions({0, 8 - dim_og % 8}));
    }
    const int dim = x.size(1);

    TORCH_CHECK(dim % 8 == 0, "fast_hadamard_transform only supports hidden dimension divisible by 8 for now");
    TORCH_CHECK(dim <= 32768, "fast_hadamard_transform only supports hidden dimension at most 32768 for now");

    at::Tensor out = torch::empty_like(x);

    HadamardParamsBase params;
    set_hadamard_params(params, batch_size, dim, 1, x, out, scale);

#if TORCH_VERSION_MAJOR > 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 6)
    c10::DeviceGuard device_guard(x.device());
#else
    at::cuda::CUDAGuard device_guard{x.device()};
#endif
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(x.scalar_type(), "fast_hadamard_transform", [&] {
        fast_hadamard_transform_cuda<input_t>(params, stream);
    });
    if (dim_og % 8 != 0) {
        out = out.index({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, dim_og)});
    }
    return out.reshape(shapes_og);
}

torch::Tensor fast_hadamard_transform_reverse(torch::Tensor x, double scale) {
    auto input_type = x.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::Float || input_type == at::ScalarType::Half || input_type == at::ScalarType::BFloat16);
    TORCH_CHECK(x.is_cuda(), "fast_hadamard_transform_reverse expects a CUDA tensor");

    const auto shapes_og = x.sizes();
    const int dim = x.size(-1);
    x = x.reshape({-1, dim});
    if (x.stride(-1) != 1) { x = x.contiguous(); }
    const int batch_size = x.size(0);

    CHECK_SHAPE(x, batch_size, dim);
    TORCH_CHECK(x.stride(1) == 1);
    TORCH_CHECK(dim >= 8 && dim <= 32768,
                "fast_hadamard_transform_reverse supports dimensions from 8 through 32768");
    TORCH_CHECK((dim & (dim - 1)) == 0,
                "fast_hadamard_transform_reverse requires a power-of-two dimension");

    at::Tensor out = torch::empty_like(x);
    HadamardParamsBase params;
    set_hadamard_params(params, batch_size, dim, 1, x, out, scale);

#if TORCH_VERSION_MAJOR > 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 6)
    c10::DeviceGuard device_guard(x.device());
#else
    at::cuda::CUDAGuard device_guard{x.device()};
#endif
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(x.scalar_type(), "fast_hadamard_transform_reverse", [&] {
        fast_hadamard_transform_reverse_cuda<input_t>(params, stream);
    });
    return out.reshape(shapes_og);
}

torch::Tensor fast_hadamard_transform_scaled(torch::Tensor x,
                                             torch::Tensor vector,
                                             double scale) {
    auto input_type = x.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::Float || input_type == at::ScalarType::Half || input_type == at::ScalarType::BFloat16);
    TORCH_CHECK(x.is_cuda(), "fast_hadamard_transform_scaled expects a CUDA tensor");

    const auto shapes_og = x.sizes();
    const int dim = x.size(-1);
    x = x.reshape({-1, dim});
    if (x.stride(-1) != 1) { x = x.contiguous(); }
    const int batch_size = x.size(0);
    TORCH_CHECK(dim >= 8 && dim <= 32768,
                "fast_hadamard_transform_scaled supports dimensions from 8 through 32768");
    TORCH_CHECK((dim & (dim - 1)) == 0,
                "fast_hadamard_transform_scaled requires a power-of-two dimension");
    TORCH_CHECK(vector.is_cuda() && vector.device() == x.device(),
                "fast_hadamard_transform_scaled requires a CUDA vector on the input device");
    TORCH_CHECK(vector.scalar_type() == input_type && vector.sizes() == torch::IntArrayRef({dim}),
                "fast_hadamard_transform_scaled vector must match the input dtype and last dimension");
    vector = vector.contiguous();

    at::Tensor out = torch::empty_like(x);
    HadamardParamsBase params;
    set_hadamard_params(params, batch_size, dim, 1, x, out, scale);
    params.vector_ptr = vector.data_ptr();

#if TORCH_VERSION_MAJOR > 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 6)
    c10::DeviceGuard device_guard(x.device());
#else
    at::cuda::CUDAGuard device_guard{x.device()};
#endif
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(input_type, "fast_hadamard_transform_scaled", [&] {
        fast_hadamard_transform_scaled_cuda<input_t>(params, stream);
    });
    return out.reshape(shapes_og);
}

torch::Tensor fast_hadamard_transform_sandwich(torch::Tensor x,
                                               torch::Tensor vector,
                                               double scale) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::ScalarType::BFloat16,
                "fast_hadamard_transform_sandwich requires a CUDA BF16 tensor");
    const auto shapes_og = x.sizes();
    const int dim = x.size(-1);
    TORCH_CHECK(dim == 512 || dim == 2048,
                "fast_hadamard_transform_sandwich currently requires dimension 512 or 2048");
    x = x.reshape({-1, dim});
    if (x.stride(-1) != 1) { x = x.contiguous(); }
    TORCH_CHECK(vector.is_cuda() && vector.device() == x.device() &&
                vector.scalar_type() == at::ScalarType::BFloat16 &&
                vector.sizes() == torch::IntArrayRef({dim}),
                "fast_hadamard_transform_sandwich requires a matching CUDA BF16 vector");
    vector = vector.contiguous();

    at::Tensor out = torch::empty_like(x);
    HadamardParamsBase params;
    set_hadamard_params(params, x.size(0), dim, 1, x, out, scale);
    params.vector_ptr = vector.data_ptr();
#if TORCH_VERSION_MAJOR > 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 6)
    c10::DeviceGuard device_guard(x.device());
#else
    at::cuda::CUDAGuard device_guard{x.device()};
#endif
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    fast_hadamard_transform_sandwich_cuda<at::BFloat16>(params, stream);
    return out.reshape(shapes_og);
}

torch::Tensor fast_hadamard_transform_reverse_scaled(torch::Tensor x,
                                                     torch::Tensor vector,
                                                     double scale) {
    auto input_type = x.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::Float || input_type == at::ScalarType::Half || input_type == at::ScalarType::BFloat16);
    TORCH_CHECK(x.is_cuda(), "fast_hadamard_transform_reverse_scaled expects a CUDA tensor");

    const auto shapes_og = x.sizes();
    const int dim = x.size(-1);
    x = x.reshape({-1, dim});
    if (x.stride(-1) != 1) { x = x.contiguous(); }
    const int batch_size = x.size(0);
    TORCH_CHECK(dim >= 8 && dim <= 32768,
                "fast_hadamard_transform_reverse_scaled supports dimensions from 8 through 32768");
    TORCH_CHECK((dim & (dim - 1)) == 0,
                "fast_hadamard_transform_reverse_scaled requires a power-of-two dimension");
    TORCH_CHECK(vector.is_cuda() && vector.device() == x.device(),
                "fast_hadamard_transform_reverse_scaled requires a CUDA vector on the input device");
    TORCH_CHECK(vector.scalar_type() == input_type && vector.sizes() == torch::IntArrayRef({dim}),
                "fast_hadamard_transform_reverse_scaled vector must match the input dtype and last dimension");
    vector = vector.contiguous();

    at::Tensor out = torch::empty_like(x);
    HadamardParamsBase params;
    set_hadamard_params(params, batch_size, dim, 1, x, out, scale);
    params.vector_ptr = vector.data_ptr();

#if TORCH_VERSION_MAJOR > 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 6)
    c10::DeviceGuard device_guard(x.device());
#else
    at::cuda::CUDAGuard device_guard{x.device()};
#endif
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(input_type, "fast_hadamard_transform_reverse_scaled", [&] {
        fast_hadamard_transform_reverse_scaled_cuda<input_t>(params, stream);
    });
    return out.reshape(shapes_og);
}

std::tuple<torch::Tensor, torch::Tensor> fast_hadamard_transform_scaled_saved(
        torch::Tensor x, torch::Tensor vector, double scale) {
    auto input_type = x.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::Float || input_type == at::ScalarType::Half || input_type == at::ScalarType::BFloat16);
    TORCH_CHECK(x.is_cuda(), "fast_hadamard_transform_scaled_saved expects a CUDA tensor");

    const auto shapes_og = x.sizes();
    const int dim = x.size(-1);
    x = x.reshape({-1, dim});
    if (x.stride(-1) != 1) { x = x.contiguous(); }
    const int batch_size = x.size(0);
    TORCH_CHECK(dim >= 8 && dim <= 32768,
                "fast_hadamard_transform_scaled_saved supports dimensions from 8 through 32768");
    TORCH_CHECK((dim & (dim - 1)) == 0,
                "fast_hadamard_transform_scaled_saved requires a power-of-two dimension");
    TORCH_CHECK(vector.is_cuda() && vector.device() == x.device(),
                "fast_hadamard_transform_scaled_saved requires a CUDA vector on the input device");
    TORCH_CHECK(vector.scalar_type() == input_type && vector.sizes() == torch::IntArrayRef({dim}),
                "fast_hadamard_transform_scaled_saved vector must match the input dtype and last dimension");
    vector = vector.contiguous();

    at::Tensor out = torch::empty_like(x);
    at::Tensor unscaled = torch::empty_like(x);
    HadamardParamsBase params;
    set_hadamard_params(params, batch_size, dim, 1, x, out, scale);
    params.vector_ptr = vector.data_ptr();
    params.auxiliary_ptr = unscaled.data_ptr();

#if TORCH_VERSION_MAJOR > 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 6)
    c10::DeviceGuard device_guard(x.device());
#else
    at::cuda::CUDAGuard device_guard{x.device()};
#endif
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(input_type, "fast_hadamard_transform_scaled_saved", [&] {
        fast_hadamard_transform_scaled_saved_cuda<input_t>(params, stream);
    });
    return {out.reshape(shapes_og), unscaled.reshape(shapes_og)};
}

std::tuple<torch::Tensor, torch::Tensor> fast_hadamard_transform_scaled_saved_bf16(
        torch::Tensor x, torch::Tensor vector, double scale) {
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Float,
                "fast_hadamard_transform_scaled_saved_bf16 requires FP32 input");
    TORCH_CHECK(x.is_cuda(), "fast_hadamard_transform_scaled_saved_bf16 expects a CUDA tensor");

    const auto shapes_og = x.sizes();
    const int dim = x.size(-1);
    x = x.reshape({-1, dim});
    if (x.stride(-1) != 1) { x = x.contiguous(); }
    const int batch_size = x.size(0);
    TORCH_CHECK(dim >= 8 && dim <= 32768,
                "fast_hadamard_transform_scaled_saved_bf16 supports dimensions from 8 through 32768");
    TORCH_CHECK((dim & (dim - 1)) == 0,
                "fast_hadamard_transform_scaled_saved_bf16 requires a power-of-two dimension");
    TORCH_CHECK(vector.is_cuda() && vector.device() == x.device(),
                "fast_hadamard_transform_scaled_saved_bf16 requires a CUDA vector on the input device");
    TORCH_CHECK(vector.scalar_type() == at::ScalarType::Float
                    && vector.sizes() == torch::IntArrayRef({dim}),
                "fast_hadamard_transform_scaled_saved_bf16 vector must be FP32 and match the last dimension");
    vector = vector.contiguous();

    at::Tensor out = torch::empty(x.sizes(), x.options().dtype(at::kBFloat16));
    at::Tensor unscaled = torch::empty_like(x);
    HadamardParamsBase params;
    set_hadamard_params(params, batch_size, dim, 1, x, out, scale);
    params.vector_ptr = vector.data_ptr();
    params.auxiliary_ptr = unscaled.data_ptr();

#if TORCH_VERSION_MAJOR > 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 6)
    c10::DeviceGuard device_guard(x.device());
#else
    at::cuda::CUDAGuard device_guard{x.device()};
#endif
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    fast_hadamard_transform_scaled_saved_cuda<float, at::BFloat16>(params, stream);
    return {out.reshape(shapes_og), unscaled.reshape(shapes_og)};
}

torch::Tensor fast_hadamard_transform_12N(torch::Tensor x, double scale) {
    auto input_type = x.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::Float || input_type == at::ScalarType::Half || input_type == at::ScalarType::BFloat16);

    TORCH_CHECK(x.is_cuda(), "fast_hadamard_transform_12N expects a CUDA tensor");

    const auto shapes_og = x.sizes();
    const int dim_og = x.size(-1);
    x = x.reshape({-1, dim_og});
    if (x.stride(-1) != 1) { x = x.contiguous(); }
    const auto sizes = x.sizes();
    const int batch_size = sizes[0];

    CHECK_SHAPE(x, batch_size, dim_og);
    TORCH_CHECK(x.stride(1) == 1);

    if (dim_og % (4 * 12) != 0) {
        x = torch::nn::functional::pad(x, torch::nn::functional::PadFuncOptions({0, (4 * 12) - dim_og % (4 * 12)}));
    }
    const int dim = x.size(1);

    TORCH_CHECK(dim % (4 * 12) == 0, "fast_hadamard_transform_12N only supports hidden dimension divisible by 48 for now");
    TORCH_CHECK(dim <= 12 * 1024, "fast_hadamard_transform_12N only supports hidden dimension at most 12288 for now");

    at::Tensor out = torch::empty_like(x);

    HadamardParamsBase params;
    set_hadamard_params(params, batch_size, dim, 12, x, out, scale);

#if TORCH_VERSION_MAJOR > 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 6)
    c10::DeviceGuard device_guard(x.device());
#else
    at::cuda::CUDAGuard device_guard{x.device()};
#endif
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(x.scalar_type(), "fast_hadamard_transform", [&] {
        fast_hadamard_transform_12N_cuda<input_t>(params, stream);
    });
    if (dim_og % (4 * 12) != 0) {
        out = out.index({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, dim_og)});
    }
    return out.reshape(shapes_og);
}

torch::Tensor fast_hadamard_transform_20N(torch::Tensor x, double scale) {
    auto input_type = x.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::Float || input_type == at::ScalarType::Half || input_type == at::ScalarType::BFloat16);

    TORCH_CHECK(x.is_cuda(), "fast_hadamard_transform_20N expects a CUDA tensor");

    const auto shapes_og = x.sizes();
    const int dim_og = x.size(-1);
    x = x.reshape({-1, dim_og});
    if (x.stride(-1) != 1) { x = x.contiguous(); }
    const auto sizes = x.sizes();
    const int batch_size = sizes[0];

    CHECK_SHAPE(x, batch_size, dim_og);
    TORCH_CHECK(x.stride(1) == 1);

    if (dim_og % (4 * 20) != 0) {
        x = torch::nn::functional::pad(x, torch::nn::functional::PadFuncOptions({0, (4 * 20) - dim_og % (4 * 20)}));
    }
    const int dim = x.size(1);

    TORCH_CHECK(dim % (4 * 20) == 0, "fast_hadamard_transform_20N only supports hidden dimension divisible by 80 for now");
    TORCH_CHECK(dim <= 20 * 1024, "fast_hadamard_transform_20N only supports hidden dimension at most 20480 for now");

    at::Tensor out = torch::empty_like(x);

    HadamardParamsBase params;
    set_hadamard_params(params, batch_size, dim, 20, x, out, scale);

#if TORCH_VERSION_MAJOR > 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 6)
    c10::DeviceGuard device_guard(x.device());
#else
    at::cuda::CUDAGuard device_guard{x.device()};
#endif
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(x.scalar_type(), "fast_hadamard_transform", [&] {
        fast_hadamard_transform_20N_cuda<input_t>(params, stream);
    });
    if (dim_og % (4 * 20) != 0) {
        out = out.index({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, dim_og)});
    }
    return out.reshape(shapes_og);
}

torch::Tensor fast_hadamard_transform_28N(torch::Tensor x, double scale) {
    auto input_type = x.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::Float || input_type == at::ScalarType::Half || input_type == at::ScalarType::BFloat16);

    TORCH_CHECK(x.is_cuda(), "fast_hadamard_transform_28N expects a CUDA tensor");

    const auto shapes_og = x.sizes();
    const int dim_og = x.size(-1);
    x = x.reshape({-1, dim_og});
    if (x.stride(-1) != 1) { x = x.contiguous(); }
    const auto sizes = x.sizes();
    const int batch_size = sizes[0];

    CHECK_SHAPE(x, batch_size, dim_og);
    TORCH_CHECK(x.stride(1) == 1);

    if (dim_og % (4 * 28) != 0) {
        x = torch::nn::functional::pad(x, torch::nn::functional::PadFuncOptions({0, (4 * 28) - dim_og % (4 * 28)}));
    }
    const int dim = x.size(1);

    TORCH_CHECK(dim % (4 * 28) == 0, "fast_hadamard_transform_28N only supports hidden dimension divisible by 112 for now");
    TORCH_CHECK(dim <= 28 * 1024, "fast_hadamard_transform_28N only supports hidden dimension at most 28672 for now");

    at::Tensor out = torch::empty_like(x);

    HadamardParamsBase params;
    set_hadamard_params(params, batch_size, dim, 28, x, out, scale);

#if TORCH_VERSION_MAJOR > 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 6)
    c10::DeviceGuard device_guard(x.device());
#else
    at::cuda::CUDAGuard device_guard{x.device()};
#endif
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(x.scalar_type(), "fast_hadamard_transform", [&] {
        fast_hadamard_transform_28N_cuda<input_t>(params, stream);
    });
    if (dim_og % (8 * 28) != 0) {
        out = out.index({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, dim_og)});
    }
    return out.reshape(shapes_og);
}

torch::Tensor fast_hadamard_transform_40N(torch::Tensor x, double scale) {
    auto input_type = x.scalar_type();
    TORCH_CHECK(input_type == at::ScalarType::Float || input_type == at::ScalarType::Half || input_type == at::ScalarType::BFloat16);

    TORCH_CHECK(x.is_cuda(), "fast_hadamard_transform_40N expects a CUDA tensor");

    const auto shapes_og = x.sizes();
    const int dim_og = x.size(-1);
    x = x.reshape({-1, dim_og});
    if (x.stride(-1) != 1) { x = x.contiguous(); }
    const auto sizes = x.sizes();
    const int batch_size = sizes[0];

    CHECK_SHAPE(x, batch_size, dim_og);
    TORCH_CHECK(x.stride(1) == 1);

    if (dim_og % (4 * 40) != 0) {
        x = torch::nn::functional::pad(x, torch::nn::functional::PadFuncOptions({0, (4 * 40) - dim_og % (4 * 40)}));
    }
    const int dim = x.size(1);

    TORCH_CHECK(dim % (4 * 40) == 0, "fast_hadamard_transform_40N only supports hidden dimension divisible by 160 for now");
    TORCH_CHECK(dim <= 40 * 1024, "fast_hadamard_transform_40N only supports hidden dimension at most 40960 for now");

    at::Tensor out = torch::empty_like(x);

    HadamardParamsBase params;
    set_hadamard_params(params, batch_size, dim, 40, x, out, scale);

#if TORCH_VERSION_MAJOR > 2 || (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 6)
    c10::DeviceGuard device_guard(x.device());
#else
    at::cuda::CUDAGuard device_guard{x.device()};
#endif
    auto stream = at::cuda::getCurrentCUDAStream().stream();
    DISPATCH_ITYPE_FLOAT_AND_HALF_AND_BF16(x.scalar_type(), "fast_hadamard_transform", [&] {
        fast_hadamard_transform_40N_cuda<input_t>(params, stream);
    });
    if (dim_og % (8 * 40) != 0) {
        out = out.index({torch::indexing::Slice(), torch::indexing::Slice(torch::indexing::None, dim_og)});
    }
    return out.reshape(shapes_og);
}

} // namespace gptqmodel_hadamard

TORCH_LIBRARY(gptqmodel_hadamard, m) {
    m.def("fast_hadamard_transform(Tensor x, float scale) -> Tensor");
    m.def("fast_hadamard_transform_reverse(Tensor x, float scale) -> Tensor");
    m.def("fast_hadamard_transform_scaled(Tensor x, Tensor vector, float scale) -> Tensor");
    m.def("fast_hadamard_transform_reverse_scaled(Tensor x, Tensor vector, float scale) -> Tensor");
    m.def("fast_hadamard_transform_scaled_saved(Tensor x, Tensor vector, float scale) -> (Tensor, Tensor)");
    m.def("fast_hadamard_transform_scaled_saved_bf16(Tensor x, Tensor vector, float scale) -> (Tensor, Tensor)");
    m.def("fast_hadamard_transform_sandwich(Tensor x, Tensor vector, float scale) -> Tensor");
    m.def("fast_hadamard_transform_12N(Tensor x, float scale) -> Tensor");
    m.def("fast_hadamard_transform_20N(Tensor x, float scale) -> Tensor");
    m.def("fast_hadamard_transform_28N(Tensor x, float scale) -> Tensor");
    m.def("fast_hadamard_transform_40N(Tensor x, float scale) -> Tensor");
    m.def("gsq_position_error(Tensor probabilities, Tensor baseline, Tensor position_indices, Tensor position_choices, Tensor position_deltas, Tensor target, Tensor(a!) error) -> ()");
}

TORCH_LIBRARY_IMPL(gptqmodel_hadamard, CUDA, m) {
    m.impl("fast_hadamard_transform", &gptqmodel_hadamard::fast_hadamard_transform);
    m.impl("fast_hadamard_transform_reverse", &gptqmodel_hadamard::fast_hadamard_transform_reverse);
    m.impl("fast_hadamard_transform_scaled", &gptqmodel_hadamard::fast_hadamard_transform_scaled);
    m.impl("fast_hadamard_transform_reverse_scaled", &gptqmodel_hadamard::fast_hadamard_transform_reverse_scaled);
    m.impl("fast_hadamard_transform_scaled_saved", &gptqmodel_hadamard::fast_hadamard_transform_scaled_saved);
    m.impl("fast_hadamard_transform_scaled_saved_bf16", &gptqmodel_hadamard::fast_hadamard_transform_scaled_saved_bf16);
    m.impl("fast_hadamard_transform_sandwich", &gptqmodel_hadamard::fast_hadamard_transform_sandwich);
    m.impl("fast_hadamard_transform_12N", &gptqmodel_hadamard::fast_hadamard_transform_12N);
    m.impl("fast_hadamard_transform_20N", &gptqmodel_hadamard::fast_hadamard_transform_20N);
    m.impl("fast_hadamard_transform_28N", &gptqmodel_hadamard::fast_hadamard_transform_28N);
    m.impl("fast_hadamard_transform_40N", &gptqmodel_hadamard::fast_hadamard_transform_40N);
    m.impl("gsq_position_error", &gptqmodel_hadamard::gsq_position_error);
}
