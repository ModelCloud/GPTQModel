// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

// Fused fast Walsh-Hadamard transform for QVQ inference on CPU.
//
// Mirrors the CUDA qvq_hadamard kernel's pre/post-scale/bias contract for
// float32 activations. The butterfly is implemented as an in-place iterative
// WHT with AVX-512 vectorization on x86-64. scale_mode 0/1 match
// matmul_hadU_stable / matmul_hadU respectively; modes 2-4 are treated as
// their FP32 equivalents (no FP16 narrowing is needed because the CPU QVQ
// path computes in float32).

#include <ATen/Parallel.h>
#include <torch/extension.h>
#include <torch/library.h>

#include <immintrin.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <cstring>
#include <limits>

namespace qvq_cpu {
namespace {

inline bool cpu_has_avx512() {
  return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
         __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512dq");
}

inline bool is_power_of_two(int64_t n) { return n > 0 && (n & (n - 1)) == 0; }

// Scalar in-place WHT.
void hadamard_row_scalar(float* buf, int n) {
  for (int stride = 1; stride < n; stride <<= 1) {
    for (int base = 0; base < n; base += 2 * stride) {
      for (int j = 0; j < stride; ++j) {
        const float a = buf[base + j];
        const float b = buf[base + j + stride];
        buf[base + j] = a + b;
        buf[base + j + stride] = a - b;
      }
    }
  }
}

// AVX-512 in-place WHT. n must be a power-of-two and at least 16.
__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx2,fma")))
void hadamard_row_avx512(float* buf, int n) {
  for (int stride = 1; stride < n; stride <<= 1) {
    if (stride < 16) {
      for (int base = 0; base < n; base += 2 * stride) {
        for (int j = 0; j < stride; ++j) {
          const float a = buf[base + j];
          const float b = buf[base + j + stride];
          buf[base + j] = a + b;
          buf[base + j + stride] = a - b;
        }
      }
    } else {
      for (int base = 0; base < n; base += 2 * stride) {
        for (int j = 0; j < stride; j += 16) {
          __m512 a = _mm512_loadu_ps(buf + base + j);
          __m512 b = _mm512_loadu_ps(buf + base + j + stride);
          __m512 sum = _mm512_add_ps(a, b);
          __m512 diff = _mm512_sub_ps(a, b);
          _mm512_storeu_ps(buf + base + j, sum);
          _mm512_storeu_ps(buf + base + j + stride, diff);
        }
      }
    }
  }
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx2,fma")))
void apply_pre_epilogue_avx512(
    float* buf,
    int n,
    const float* pre_scale,
    bool normalize_first,
    float norm) {
  if (pre_scale) {
    for (int i = 0; i < n; i += 16) {
      __m512 v = _mm512_loadu_ps(buf + i);
      v = _mm512_mul_ps(v, _mm512_loadu_ps(pre_scale + i));
      if (normalize_first) v = _mm512_div_ps(v, _mm512_set1_ps(norm));
      _mm512_storeu_ps(buf + i, v);
    }
  } else if (normalize_first) {
    __m512 nvec = _mm512_set1_ps(norm);
    for (int i = 0; i < n; i += 16) {
      __m512 v = _mm512_loadu_ps(buf + i);
      v = _mm512_div_ps(v, nvec);
      _mm512_storeu_ps(buf + i, v);
    }
  }
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx2,fma")))
void apply_post_epilogue_avx512(
    float* buf,
    int n,
    bool normalize_last,
    float norm,
    const float* post_scale,
    const float* bias) {
  __m512 nvec = _mm512_set1_ps(norm);
  if (post_scale && bias) {
    for (int i = 0; i < n; i += 16) {
      __m512 v = _mm512_loadu_ps(buf + i);
      if (normalize_last) v = _mm512_div_ps(v, nvec);
      v = _mm512_mul_ps(v, _mm512_loadu_ps(post_scale + i));
      v = _mm512_add_ps(v, _mm512_loadu_ps(bias + i));
      _mm512_storeu_ps(buf + i, v);
    }
  } else if (post_scale) {
    for (int i = 0; i < n; i += 16) {
      __m512 v = _mm512_loadu_ps(buf + i);
      if (normalize_last) v = _mm512_div_ps(v, nvec);
      v = _mm512_mul_ps(v, _mm512_loadu_ps(post_scale + i));
      _mm512_storeu_ps(buf + i, v);
    }
  } else if (bias) {
    for (int i = 0; i < n; i += 16) {
      __m512 v = _mm512_loadu_ps(buf + i);
      if (normalize_last) v = _mm512_div_ps(v, nvec);
      v = _mm512_add_ps(v, _mm512_loadu_ps(bias + i));
      _mm512_storeu_ps(buf + i, v);
    }
  } else if (normalize_last) {
    for (int i = 0; i < n; i += 16) {
      __m512 v = _mm512_loadu_ps(buf + i);
      v = _mm512_div_ps(v, nvec);
      _mm512_storeu_ps(buf + i, v);
    }
  }
}

void apply_pre_epilogue_scalar(
    float* buf,
    int n,
    const float* pre_scale,
    bool normalize_first,
    float norm) {
  if (pre_scale) {
    if (normalize_first) {
      for (int i = 0; i < n; ++i) buf[i] = pre_scale[i] * buf[i] / norm;
    } else {
      for (int i = 0; i < n; ++i) buf[i] = pre_scale[i] * buf[i];
    }
  } else if (normalize_first) {
    for (int i = 0; i < n; ++i) buf[i] /= norm;
  }
}

void apply_post_epilogue_scalar(
    float* buf,
    int n,
    bool normalize_last,
    float norm,
    const float* post_scale,
    const float* bias) {
  if (post_scale) {
    if (bias) {
      if (normalize_last) {
        for (int i = 0; i < n; ++i) buf[i] = buf[i] / norm * post_scale[i] + bias[i];
      } else {
        for (int i = 0; i < n; ++i) buf[i] = buf[i] * post_scale[i] + bias[i];
      }
    } else {
      if (normalize_last) {
        for (int i = 0; i < n; ++i) buf[i] = buf[i] / norm * post_scale[i];
      } else {
        for (int i = 0; i < n; ++i) buf[i] *= post_scale[i];
      }
    }
  } else if (bias) {
    if (normalize_last) {
      for (int i = 0; i < n; ++i) buf[i] = buf[i] / norm + bias[i];
    } else {
      for (int i = 0; i < n; ++i) buf[i] += bias[i];
    }
  } else if (normalize_last) {
    for (int i = 0; i < n; ++i) buf[i] /= norm;
  }
}

}  // namespace

at::Tensor qvq_hadamard_cpu(
    const at::Tensor& input,
    const c10::optional<at::Tensor>& pre_scale,
    const c10::optional<at::Tensor>& post_scale,
    const c10::optional<at::Tensor>& bias,
    int64_t scale_mode,
    bool pad_to_16,
    bool output_fp16,
    bool output_bf16,
    const c10::optional<at::Tensor>& input_scale,
    int64_t input_rounding_mode) {
  TORCH_CHECK(input.device().is_cpu(), "qvq_hadamard_cpu: input must be a CPU tensor");
  TORCH_CHECK(input.dim() >= 1, "qvq_hadamard_cpu: input must be at least rank one");
  TORCH_CHECK(input.scalar_type() == at::kFloat, "qvq_hadamard_cpu: only float32 input is supported");
  TORCH_CHECK(input.is_contiguous(), "qvq_hadamard_cpu: input must be contiguous");
  TORCH_CHECK(scale_mode >= 0 && scale_mode <= 4, "qvq_hadamard_cpu: scale_mode must be in [0, 4]");
  TORCH_CHECK(!pad_to_16, "qvq_hadamard_cpu: padded M16 output is CUDA-only");
  TORCH_CHECK(!output_fp16, "qvq_hadamard_cpu: FP16 output is CUDA-only");
  TORCH_CHECK(!output_bf16, "qvq_hadamard_cpu: BF16 output is CUDA-only");
  TORCH_CHECK(!input_scale.has_value() || !input_scale->defined(),
              "qvq_hadamard_cpu: scaled FP8 input is CUDA-only");
  TORCH_CHECK(input_rounding_mode == 0,
              "qvq_hadamard_cpu: input_rounding_mode requires scaled FP8 input");

  int64_t n = input.size(-1);
  TORCH_CHECK(n >= 2 && is_power_of_two(n), "qvq_hadamard_cpu: last dim must be a power-of-two >= 2");
  TORCH_CHECK(n <= 16384, "qvq_hadamard_cpu: last dim must be <= 16384");

  const auto check_optional = [&](const c10::optional<at::Tensor>& t, const char* name) {
    if (t.has_value() && t->defined()) {
      TORCH_CHECK(t->is_cpu(), "qvq_hadamard_cpu: ", name, " must be a CPU tensor");
      TORCH_CHECK(t->scalar_type() == at::kFloat, "qvq_hadamard_cpu: ", name, " must be float32");
      TORCH_CHECK(t->is_contiguous(), "qvq_hadamard_cpu: ", name, " must be contiguous");
      TORCH_CHECK(t->numel() == n, "qvq_hadamard_cpu: ", name, " must have ", n, " elements");
    }
  };
  check_optional(pre_scale, "pre_scale");
  check_optional(post_scale, "post_scale");
  check_optional(bias, "bias");

  int64_t rows = input.numel() / n;
  if (rows == 0) {
    return at::empty_like(input);
  }
  TORCH_CHECK(rows <= std::numeric_limits<int>::max(), "qvq_hadamard_cpu: row count exceeds int limit");

  const float* in_ptr = input.data_ptr<float>();
  at::Tensor output = at::empty_like(input);
  float* out_ptr = output.data_ptr<float>();

  const float* pre_ptr = pre_scale.has_value() && pre_scale->defined() ? pre_scale->data_ptr<float>() : nullptr;
  const float* post_ptr = post_scale.has_value() && post_scale->defined() ? post_scale->data_ptr<float>() : nullptr;
  const float* bias_ptr = bias.has_value() && bias->defined() ? bias->data_ptr<float>() : nullptr;

  const bool normalize_first = scale_mode == 0 || scale_mode == 2 || scale_mode == 3;
  const bool normalize_last = scale_mode == 1 || scale_mode == 4;
  const float norm = std::sqrt(static_cast<float>(n));
  const bool use_avx512 = cpu_has_avx512() && n >= 16;

  at::parallel_for(0, rows, 1, [&](int64_t begin, int64_t end) {
    // Per-thread scratch buffer for one row.
    alignas(64) static thread_local float row_buf[16384];
    for (int64_t r = begin; r < end; ++r) {
      const float* src = in_ptr + r * n;
      float* dst = out_ptr + r * n;
      std::memcpy(row_buf, src, n * sizeof(float));

      if (use_avx512) {
        apply_pre_epilogue_avx512(row_buf, static_cast<int>(n), pre_ptr, normalize_first, norm);
        hadamard_row_avx512(row_buf, static_cast<int>(n));
        apply_post_epilogue_avx512(row_buf, static_cast<int>(n), normalize_last, norm, post_ptr, bias_ptr);
      } else {
        apply_pre_epilogue_scalar(row_buf, static_cast<int>(n), pre_ptr, normalize_first, norm);
        hadamard_row_scalar(row_buf, static_cast<int>(n));
        apply_post_epilogue_scalar(row_buf, static_cast<int>(n), normalize_last, norm, post_ptr, bias_ptr);
      }

      std::memcpy(dst, row_buf, n * sizeof(float));
    }
  });

  return output;
}

}  // namespace qvq_cpu



namespace {

// torch rejects duplicate def() calls even for byte-identical schemas, and
// cross-bundle schema lookups are unreliable during JIT-plugin static init.
// The CPU and CUDA QVQ extensions intentionally share these op schemas, so
// each def runs at most once per process: a duplicate registration throws
// c10::Error before mutating dispatcher state and is swallowed here.
template <typename DefFn>
void qvq_def_shared_schema(DefFn&& def_fn) {
  try {
    def_fn();
  } catch (const std::exception&) {
  }
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(gptqmodel_qvq, m) {
  qvq_def_shared_schema([&] {
    m.def("hadamard(Tensor input, Tensor? pre_scale, Tensor? post_scale, Tensor? bias, int scale_mode, bool pad_to_16=False, bool output_fp16=False, bool output_bf16=False, Tensor? input_scale=None, int input_rounding_mode=0) -> Tensor");
});
}

TORCH_LIBRARY_IMPL(gptqmodel_qvq, CPU, m) {
  m.impl("hadamard", qvq_cpu::qvq_hadamard_cpu);
}
