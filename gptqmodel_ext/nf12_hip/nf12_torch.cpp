// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-FileCopyrightText: 2026 AlpinDale and the nf12 contributors
// SPDX-License-Identifier: Apache-2.0
// ---------------------------------------------------------------------------
// Torch bindings for the NF12 HIP port (AMD RDNA4: gfx1200/gfx1201/gfx1250).
//
// Credits: NyoomFloat12 algorithm by Liberman, Fang, Zhang, Dao and Fu
// (Together AI, 2026), https://www.together.ai/. NF12 format, reference codec
// and CUDA kernels by AlpinDale, https://github.com/dphnAI/nf12 (Apache-2.0).
//
// Registers `torch.ops.nf12_hip` with the same decode/linear contract as
// the CUDA `nf12` namespace so the Python layer routes by device family
// without changing callers.
//
// Host-only translation unit: the ROCm SDK here ships no thrust, so the .hip
// files must not see torch/ATen. Mirrors allspark_torch_hip.cpp.
#include <torch/extension.h>
#include <torch/library.h>

#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/util/SmallVector.h>
#include <ATen/ops/empty_native.h>
#include <ATen/cuda/CUDAContext.h>

#include <hip/hip_runtime_api.h>

#include <optional>
#include <atomic>
#include <limits>
#include <vector>
#include <string>

// ---------------------------------------------------------------------------
// Device-side entry points (nf12_hip.hip). No torch types cross this line.
// ---------------------------------------------------------------------------
extern "C" {

// `out` is deliberately `void*`: the .hip translation unit owns the bf16 type
// (__hip_bfloat16) and hipify would mangle any HIP bf16 spelling used here.
hipError_t nf12_decode(const uint8_t* packed, const uint4* escapes, uint32_t exp_base,
                          int64_t n, int64_t k, void* out, int64_t ldo,
                          hipStream_t stream);

hipError_t nf12_decode_fp32(const uint8_t* packed, const uint4* escapes, uint32_t exp_base,
                            int64_t n, int64_t k, void* out, int64_t ldo,
                            hipStream_t stream);

hipError_t nf12_sub_fp32(const uint8_t* packed, const uint4* escapes, uint32_t exp_base,
                         int64_t n, int64_t k, const void* wq, void* out, int64_t ldo,
                         hipStream_t stream);

hipError_t nf12_embedding(const uint8_t* packed, const uint4* escapes,
                          uint32_t num_escapes, uint32_t exp_base,
                          int64_t n, int64_t k, const int32_t* ids32,
                          const int64_t* ids64, int64_t num_ids, void* out,
                          hipStream_t stream);

hipError_t nf12_rowmajor_count(const uint8_t* source, int64_t n, int64_t k,
                               int32_t* counts, hipStream_t stream);
hipError_t nf12_rowmajor_pack(const uint8_t* source,
                              const uint4* original_escapes,
                              int64_t original_count, int64_t n, int64_t k,
                              uint32_t exp_base, const int64_t* prefix,
                              uint8_t* dest, uint64_t* dest_escapes,
                              int32_t* invalid_escape, hipStream_t stream);
hipError_t nf12_rowmajor_lookup(const void* ids, const uint8_t* packed,
                                const uint64_t* escapes,
                                uint32_t num_escapes, uint32_t exp_base,
                                int64_t n, int64_t k, int64_t num_ids,
                                void* out, int id_bytes, hipStream_t stream);

hipError_t nf12_selftest(uint32_t* out, hipStream_t stream);

// Fused small-M decode+GEMM. `x` and `out` are BF16, row-major, [m, k] and
// [m, n]. `out` may be `void*` for the same reason as decode's.
int nf12_linear_fused_small_m_supported(int64_t m);
hipError_t nf12_linear_fused(const uint8_t* packed, const uint4* escapes, uint32_t exp_base,
                                const void* x, void* out, int64_t n, int64_t k, int64_t m,
                                const void* bias, hipStream_t stream);

// Matrix-core path for the mid-M band (nf12_wmma.hip). Returns 0 when the
// device has no wmma-128b-insts (gfx1250), so callers keep decode-then-GEMM.
int nf12_wmma_supported(int64_t m);
size_t nf12_wmma_workspace_bytes(int64_t n, int64_t k, int64_t m);
hipError_t nf12_linear_wmma(const uint8_t* packed, const uint4* escapes, uint32_t exp_base,
                               const void* x, void* out, int64_t n, int64_t k, int64_t m,
                               const void* bias, void* workspace, size_t workspace_bytes,
                               hipStream_t stream);

// Encode (nf12_encode.hip). Two phases so escapes are allocated exactly.
size_t nf12_encode_workspace_bytes(int64_t n, int64_t k);
hipError_t nf12_encode_analyze(const void* w, int64_t n, int64_t k, int64_t ldw,
                                  void* workspace, size_t workspace_bytes, uint8_t* exp_base_out,
                                  uint32_t* num_escapes_out, hipStream_t stream);
hipError_t nf12_encode(const void* w, int64_t n, int64_t k, int64_t ldw, uint8_t exp_base,
                          uint8_t* packed, uint4* escapes, uint32_t escapes_capacity,
                          void* workspace, size_t workspace_bytes, hipStream_t stream);
}

namespace {

constexpr int64_t kMaxExpBase = 120;

inline hipStream_t current_stream() {
  return reinterpret_cast<hipStream_t>(c10::cuda::getCurrentCUDAStream().stream());
}

inline c10::cuda::CUDAGuard device_guard(const torch::Tensor& t) {
  return c10::cuda::CUDAGuard(t.device());
}

inline std::string arch_name(int device) {
  hipDeviceProp_t props;
  TORCH_CHECK(hipGetDeviceProperties(&props, device) == hipSuccess,
              "nf12_hip: could not query device properties");
  const std::string full(props.gcnArchName);
  return full.substr(0, full.find(':'));
}

inline void check_arch(const torch::Tensor& t) {
  TORCH_CHECK(t.is_cuda(), "nf12_hip requires a CUDA/HIP tensor");
  constexpr int kCachedDevices = 64;
  static std::atomic<uint8_t> supported[kCachedDevices];
  const int device = t.get_device();
  if (device >= 0 && device < kCachedDevices &&
      supported[device].load(std::memory_order_acquire) != 0) return;

  const std::string arch = arch_name(device);
  const bool is_supported =
      arch == "gfx1200" || arch == "gfx1201" || arch == "gfx1250";
  TORCH_CHECK(arch == "gfx1200" || arch == "gfx1201" || arch == "gfx1250",
              "nf12_hip requires RDNA4 (gfx1200/gfx1201/gfx1250), got ", arch);
  if (device >= 0 && device < kCachedDevices && is_supported) {
    supported[device].store(1, std::memory_order_release);
  }
}

// ---------------------------------------------------------------------------
// nf12_hip::decode(packed, escapes, exp_base) -> bf16 [N, K]
// nf12_hip::decode_fp32(packed, escapes, exp_base) -> fp32 [N, K]
// nf12_hip::sub_fp32(packed, escapes, exp_base, wq) -> fp32 [N, K]
// ---------------------------------------------------------------------------
std::pair<int64_t, int64_t> decode_geometry(const torch::Tensor& packed,
                                            const torch::Tensor& escapes,
                                            int64_t exp_base, const char* op) {
  check_arch(packed);
  TORCH_CHECK(packed.dim() == 2 && packed.scalar_type() == at::kByte,
              op, ": packed must be a 2-D uint8 tensor");
  TORCH_CHECK(packed.is_contiguous(), op, ": packed must be contiguous");
  TORCH_CHECK(escapes.scalar_type() == at::kShort && escapes.dim() == 2 &&
                  escapes.size(1) == 8,
              op, ": escapes must be int16 [E, 8]");
  TORCH_CHECK(escapes.is_contiguous(), op, ": escapes must be contiguous");
  TORCH_CHECK(packed.device() == escapes.device(),
              op, ": packed and escapes must share a device");
  TORCH_CHECK(exp_base >= 0 && exp_base <= kMaxExpBase,
              op, ": exp_base must be in [0, 120], got ", exp_base);

  const int64_t n = packed.size(0);
  const int64_t row_bytes = packed.size(1);
  TORCH_CHECK(n > 0 && n % 16 == 0,
              op, ": N must be a positive multiple of 16");
  TORCH_CHECK(row_bytes % 96 == 0,
              op, ": packed row must be a multiple of 96 bytes");
  const int64_t k = row_bytes * 2 / 3;
  TORCH_CHECK(k % 64 == 0, op, ": K must be a multiple of 64");
  return {n, k};
}

torch::Tensor decode_common(const torch::Tensor& packed, const torch::Tensor& escapes,
                            int64_t exp_base, at::ScalarType dtype, const char* op) {
  const auto [n, k] = decode_geometry(packed, escapes, exp_base, op);

  auto guard = device_guard(packed);
  auto out = torch::empty({n, k}, packed.options().dtype(dtype));

  const auto* escapes_ptr = reinterpret_cast<const uint4*>(escapes.data_ptr<int16_t>());
  const hipError_t status = dtype == at::kFloat
      ? nf12_decode_fp32(packed.data_ptr<uint8_t>(), escapes_ptr,
                         static_cast<uint32_t>(exp_base), n, k,
                         static_cast<void*>(out.data_ptr()), k, current_stream())
      : nf12_decode(packed.data_ptr<uint8_t>(), escapes_ptr,
                    static_cast<uint32_t>(exp_base), n, k,
                    static_cast<void*>(out.data_ptr()), k, current_stream());
  TORCH_CHECK(status == hipSuccess, op, " failed: ", hipGetErrorString(status));
  return out;
}

torch::Tensor decode_dispatch(const torch::Tensor& packed, const torch::Tensor& escapes,
                              int64_t exp_base) {
  return decode_common(packed, escapes, exp_base, at::kBFloat16, "nf12_hip.decode");
}

torch::Tensor decode_fp32_dispatch(const torch::Tensor& packed, const torch::Tensor& escapes,
                                   int64_t exp_base) {
  return decode_common(packed, escapes, exp_base, at::kFloat, "nf12_hip.decode_fp32");
}

torch::Tensor sub_fp32_dispatch(const torch::Tensor& packed, const torch::Tensor& escapes,
                                int64_t exp_base, const torch::Tensor& wq) {
  const char* op = "nf12_hip.sub_fp32";
  const auto [n, k] = decode_geometry(packed, escapes, exp_base, op);
  TORCH_CHECK(wq.scalar_type() == at::kFloat && wq.dim() == 2 && wq.size(0) == n &&
                  wq.size(1) == k,
              op, ": wq must be a float32 [N, K] matrix");
  TORCH_CHECK(wq.device() == packed.device(), op, ": wq must share the packed device");
  TORCH_CHECK(wq.is_contiguous(), op, ": wq must be contiguous");

  auto guard = device_guard(packed);
  auto out = torch::empty({n, k}, packed.options().dtype(at::kFloat));
  const hipError_t status = nf12_sub_fp32(
      packed.data_ptr<uint8_t>(),
      reinterpret_cast<const uint4*>(escapes.data_ptr<int16_t>()),
      static_cast<uint32_t>(exp_base), n, k,
      static_cast<const void*>(wq.data_ptr()),
      static_cast<void*>(out.data_ptr()), k, current_stream());
  TORCH_CHECK(status == hipSuccess, op, " failed: ", hipGetErrorString(status));
  return out;
}

// ---------------------------------------------------------------------------
// nf12_hip::embedding(ids, packed, escapes, exp_base) -> bf16 [*ids.shape, K]
// ---------------------------------------------------------------------------
torch::Tensor embedding_dispatch(const torch::Tensor& ids_in,
                                 const torch::Tensor& packed,
                                 const torch::Tensor& escapes,
                                 int64_t exp_base) {
  check_arch(packed);
  TORCH_CHECK(ids_in.is_cuda() && ids_in.device() == packed.device(),
              "nf12_hip.embedding: ids and packed must share a CUDA/HIP device");
  TORCH_CHECK(ids_in.scalar_type() == at::kInt || ids_in.scalar_type() == at::kLong,
              "nf12_hip.embedding: ids must be int32 or int64");
  TORCH_CHECK(packed.dim() == 2 && packed.scalar_type() == at::kByte && packed.is_contiguous(),
              "nf12_hip.embedding: packed must be contiguous uint8 [N, 3*K/2]");
  TORCH_CHECK(escapes.dim() == 2 && escapes.size(1) == 8 &&
                  escapes.scalar_type() == at::kShort && escapes.is_contiguous() &&
                  escapes.device() == packed.device(),
              "nf12_hip.embedding: escapes must be contiguous int16 [E, 8] on the packed device");
  TORCH_CHECK(exp_base >= 0 && exp_base <= kMaxExpBase,
              "nf12_hip.embedding: exp_base must be in [0, 120]");
  const int64_t n = packed.size(0);
  const int64_t row_bytes = packed.size(1);
  TORCH_CHECK(n > 0 && n % 16 == 0 && row_bytes > 0 && row_bytes % 96 == 0,
              "nf12_hip.embedding: N must be a positive multiple of 16 and K a positive multiple of 64");
  const int64_t k = (row_bytes / 3) * 2;
  TORCH_CHECK(escapes.size(0) <= std::numeric_limits<uint32_t>::max(),
              "nf12_hip.embedding: escape table is too large");

  auto guard = device_guard(packed);
  const torch::Tensor ids = ids_in.is_contiguous() ? ids_in : ids_in.contiguous();
  c10::SmallVector<int64_t, 5> out_sizes(ids.sizes().begin(), ids.sizes().end());
  out_sizes.push_back(k);
  auto out = at::native::empty_cuda(
      out_sizes, at::kBFloat16, std::nullopt, packed.device(), false, std::nullopt);
  const int64_t num_ids = ids.numel();
  if (num_ids == 0) return out;

  const hipError_t status = nf12_embedding(
      packed.data_ptr<uint8_t>(),
      escapes.numel() ? reinterpret_cast<const uint4*>(escapes.data_ptr<int16_t>()) : nullptr,
      static_cast<uint32_t>(escapes.size(0)), static_cast<uint32_t>(exp_base),
      n, k,
      ids.scalar_type() == at::kInt ? ids.data_ptr<int32_t>() : nullptr,
      ids.scalar_type() == at::kLong ? ids.data_ptr<int64_t>() : nullptr,
      num_ids, static_cast<void*>(out.data_ptr()), current_stream());
  TORCH_CHECK(status == hipSuccess, "nf12_hip.embedding failed: ", hipGetErrorString(status));
  return out;
}

// ---------------------------------------------------------------------------
// nf12_hip::rowmajor_pack(packed, escapes, exp_base)
//     -> (row_packed, row_escapes, original_escapes, collisions)
// ---------------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor, int64_t, int64_t>
rowmajor_pack_dispatch(const torch::Tensor& packed,
                       const torch::Tensor& escapes,
                       int64_t exp_base) {
  check_arch(packed);
  TORCH_CHECK(packed.dim() == 2 && packed.scalar_type() == at::kByte &&
                  packed.is_contiguous(),
              "nf12_hip.rowmajor_pack: packed must be contiguous uint8 [N, 3*K/2]");
  TORCH_CHECK(escapes.dim() == 2 && escapes.size(1) == 8 &&
                  escapes.scalar_type() == at::kShort && escapes.is_contiguous() &&
                  escapes.device() == packed.device(),
              "nf12_hip.rowmajor_pack: escapes must be contiguous int16 [E, 8]");
  TORCH_CHECK(exp_base >= 0 && exp_base <= kMaxExpBase,
              "nf12_hip.rowmajor_pack: exp_base must be in [0, 120]");
  const int64_t n = packed.size(0);
  const int64_t row_bytes = packed.size(1);
  TORCH_CHECK(n > 0 && n % 16 == 0 && row_bytes > 0 && row_bytes % 96 == 0,
              "nf12_hip.rowmajor_pack: N must be a positive multiple of 16 and K a positive multiple of 64");
  const int64_t k = (row_bytes / 3) * 2;
  TORCH_CHECK(escapes.size(0) <= std::numeric_limits<uint32_t>::max(),
              "nf12_hip.rowmajor_pack: escape table is too large");

  auto guard = device_guard(packed);
  const hipStream_t stream = current_stream();
  const int64_t tiles = (n / 16) * (k / 64);
  TORCH_CHECK(tiles > 0 && tiles <= std::numeric_limits<int32_t>::max(),
              "nf12_hip.rowmajor_pack: tile count exceeds int32 capacity");
  auto counts = torch::empty({tiles}, packed.options().dtype(at::kInt));
  counts.zero_();
  hipError_t status = nf12_rowmajor_count(
      packed.data_ptr<uint8_t>(), n, k, counts.data_ptr<int32_t>(), stream);
  TORCH_CHECK(status == hipSuccess,
              "nf12_hip.rowmajor_pack count failed: ", hipGetErrorString(status));

  const torch::Tensor prefix = counts.cumsum(0, at::kLong);
  const int64_t collisions = prefix.select(0, tiles - 1).item<int64_t>();
  const int64_t original = escapes.size(0);
  TORCH_CHECK(collisions >= 0 &&
                  original <= (std::numeric_limits<int64_t>::max() - collisions) / 2,
              "nf12_hip.rowmajor_pack: escape table overflow");
  const int64_t total_escapes = 2 * original + collisions;
  TORCH_CHECK(total_escapes <= std::numeric_limits<uint32_t>::max(),
              "nf12_hip.rowmajor_pack: escape index exceeds uint32 capacity");

  auto row_packed =
      torch::empty({n, k * 3 / 2}, packed.options().dtype(at::kByte));
  auto row_escapes =
      torch::empty({total_escapes, 4}, packed.options().dtype(at::kShort));
  auto invalid = torch::zeros({1}, packed.options().dtype(at::kInt));
  status = nf12_rowmajor_pack(
      packed.data_ptr<uint8_t>(),
      original != 0
          ? reinterpret_cast<const uint4*>(escapes.data_ptr<int16_t>())
          : nullptr,
      original, n, k, static_cast<uint32_t>(exp_base),
      prefix.data_ptr<int64_t>(), row_packed.data_ptr<uint8_t>(),
      total_escapes != 0
          ? reinterpret_cast<uint64_t*>(row_escapes.data_ptr<int16_t>())
          : nullptr,
      invalid.data_ptr<int32_t>(), stream);
  TORCH_CHECK(status == hipSuccess,
              "nf12_hip.rowmajor_pack failed: ", hipGetErrorString(status));
  TORCH_CHECK(invalid.item<int32_t>() == 0,
              "nf12_hip.rowmajor_pack: canonical escape index out of range");
  return {row_packed, row_escapes, original, collisions};
}

// ---------------------------------------------------------------------------
// nf12_hip::rowmajor_embedding(ids, row_packed, row_escapes, exp_base)
// ---------------------------------------------------------------------------
torch::Tensor rowmajor_embedding_dispatch(const torch::Tensor& ids_in,
                                          const torch::Tensor& packed,
                                          const torch::Tensor& escapes,
                                          int64_t exp_base) {
  check_arch(packed);
  TORCH_CHECK(ids_in.is_cuda() && ids_in.device() == packed.device(),
              "nf12_hip.rowmajor_embedding: ids and packed must share a CUDA/HIP device");
  TORCH_CHECK(ids_in.scalar_type() == at::kInt || ids_in.scalar_type() == at::kLong,
              "nf12_hip.rowmajor_embedding: ids must be int32 or int64");
  TORCH_CHECK(packed.dim() == 2 && packed.scalar_type() == at::kByte &&
                  packed.is_contiguous(),
              "nf12_hip.rowmajor_embedding: packed must be contiguous uint8 [N, 3*K/2]");
  TORCH_CHECK(escapes.dim() == 2 && escapes.size(1) == 4 &&
                  escapes.scalar_type() == at::kShort && escapes.is_contiguous() &&
                  escapes.device() == packed.device(),
              "nf12_hip.rowmajor_embedding: escapes must be contiguous int16 [E, 4]");
  TORCH_CHECK(exp_base >= 0 && exp_base <= kMaxExpBase,
              "nf12_hip.rowmajor_embedding: exp_base must be in [0, 120]");
  const int64_t n = packed.size(0);
  const int64_t row_bytes = packed.size(1);
  TORCH_CHECK(n > 0 && n % 16 == 0 && row_bytes > 0 && row_bytes % 96 == 0,
              "nf12_hip.rowmajor_embedding: N must be a positive multiple of 16 and K a positive multiple of 64");
  const int64_t k = (row_bytes / 3) * 2;
  TORCH_CHECK(escapes.size(0) <= std::numeric_limits<uint32_t>::max(),
              "nf12_hip.rowmajor_embedding: escape table is too large");

  auto guard = device_guard(packed);
  const torch::Tensor ids = ids_in.is_contiguous() ? ids_in : ids_in.contiguous();
  c10::SmallVector<int64_t, 5> out_sizes(ids.sizes().begin(), ids.sizes().end());
  out_sizes.push_back(k);
  auto out = at::native::empty_cuda(
      out_sizes, at::kBFloat16, std::nullopt, packed.device(), false, std::nullopt);
  const int64_t num_ids = ids.numel();
  if (num_ids == 0) return out;

  const hipError_t status = nf12_rowmajor_lookup(
      ids.data_ptr(),
      packed.data_ptr<uint8_t>(),
      escapes.numel()
          ? reinterpret_cast<const uint64_t*>(escapes.data_ptr<int16_t>())
          : nullptr,
      static_cast<uint32_t>(escapes.size(0)), static_cast<uint32_t>(exp_base),
      n, k, num_ids, static_cast<void*>(out.data_ptr()),
      static_cast<int>(ids.element_size()), current_stream());
  TORCH_CHECK(status == hipSuccess,
              "nf12_hip.rowmajor_embedding failed: ", hipGetErrorString(status));
  return out;
}

// ---------------------------------------------------------------------------
// nf12_hip::encode(w) -> (packed, escapes, exp_base)
//
// Same two-phase contract as the CUDA `nf12.encode`: analyze picks exp_base
// and the exact escape count, then encode writes packed/escapes. Output is
// byte-identical to the Torch codec oracle (asserted by the validator).
// ---------------------------------------------------------------------------
std::tuple<torch::Tensor, torch::Tensor, int64_t> encode_dispatch(const torch::Tensor& w_in) {
  check_arch(w_in);
  TORCH_CHECK(w_in.is_cuda(), "nf12_hip.encode: weight must be a CUDA tensor");
  TORCH_CHECK(w_in.scalar_type() == at::kBFloat16,
              "nf12_hip.encode: weight must be bfloat16");
  TORCH_CHECK(w_in.dim() == 2, "nf12_hip.encode: weight must be 2-D");
  const torch::Tensor w = w_in.contiguous();
  const int64_t n = w.size(0);
  const int64_t k = w.size(1);
  TORCH_CHECK(n > 0 && n % 16 == 0 && k > 0 && k % 64 == 0,
              "nf12_hip.encode: needs N % 16 == 0 and K % 64 == 0, got N=", n, " K=", k);

  auto guard = device_guard(w);
  const hipStream_t stream = current_stream();
  const size_t ws_bytes = nf12_encode_workspace_bytes(n, k);
  auto ws = torch::empty({static_cast<int64_t>(ws_bytes)},
                         w.options().dtype(at::kByte));

  uint8_t exp_base = 0;
  uint32_t num_escapes = 0;
  hipError_t status = nf12_encode_analyze(
      static_cast<const void*>(w.data_ptr()), n, k, w.stride(0),
      static_cast<void*>(ws.data_ptr()), ws_bytes, &exp_base, &num_escapes, stream);
  TORCH_CHECK(status == hipSuccess, "nf12_hip.encode (analyze) failed: ",
              hipGetErrorString(status));

  auto packed = torch::empty({n, k * 3 / 2}, w.options().dtype(at::kByte));
  auto escapes = torch::empty({static_cast<int64_t>(num_escapes), 8},
                              w.options().dtype(at::kShort));
  status = nf12_encode(
      static_cast<const void*>(w.data_ptr()), n, k, w.stride(0), exp_base,
      packed.data_ptr<uint8_t>(),
      num_escapes > 0 ? reinterpret_cast<uint4*>(escapes.data_ptr<int16_t>()) : nullptr,
      num_escapes, static_cast<void*>(ws.data_ptr()), ws_bytes, stream);
  TORCH_CHECK(status == hipSuccess, "nf12_hip.encode failed: ", hipGetErrorString(status));
  return {packed, escapes, static_cast<int64_t>(exp_base)};
}

// ---------------------------------------------------------------------------
// nf12_hip::linear(x, packed, escapes, exp_base, bias) -> bf16
// ---------------------------------------------------------------------------
// SPEC.md 6.5 decode-then-GEMM: decode once into a caching-allocator buffer,
// then one BF16 GEMM. The buffer is a local temporary owned by this call; no
// dense weight is cached on the module (AGENTS.md quantized-storage rule).
torch::Tensor linear_dispatch(const torch::Tensor& x, const torch::Tensor& packed,
                              const torch::Tensor& escapes, int64_t exp_base,
                              const std::optional<torch::Tensor>& bias) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16,
              "nf12_hip.linear: x must be a bf16 CUDA tensor");
  TORCH_CHECK(packed.is_cuda() && x.device() == packed.device() &&
                  escapes.device() == packed.device(),
              "nf12_hip.linear: x, packed and escapes must share a CUDA/HIP device");
  // wmma_supported queries the current HIP device, and current_stream below
  // must belong to the device that owns the packed weight.
  auto guard = device_guard(packed);
  const int64_t k = x.size(-1);
  const int64_t n = packed.size(0);
  TORCH_CHECK(packed.size(1) * 2 / 3 == k, "nf12_hip.linear: x's last dim (", k,
              ") must equal K (", packed.size(1) * 2 / 3, ")");

  // Prefer the fused small-M kernel: it reads only the packed weight
  // (K*3/2 bytes per row) and keeps the decoded super-atom in registers.
  // Decode-then-GEMM below moves packed + dense write + dense read, which at
  // M <= 15 is dominated by materialising a matrix the GEMM immediately
  // re-reads, so it cannot reach BF16 parity. Large M still wants the GEMM.
  //
  // m == 16 is exactly one matrix-core tile and measured faster than the fused
  // kernel on every shape (see the RDNA4 README table), so the fused route
  // stops at 15 and 16 joins the wmma route below.
  const int64_t m = x.numel() / k;
  // Every route below indexes rows of a contiguous [m, k] matrix, so a sliced
  // or padded activation (a legal torch layout) must be normalized here. The
  // kernels cannot honour a wider row stride: forwarding it silently read the
  // wrong rows. The output keeps the caller's leading dimensions.
  const torch::Tensor x2 = x.contiguous().reshape({m, k});
  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  out_sizes.back() = n;
  if (m < 16 && nf12_linear_fused_small_m_supported(m)) {
    check_arch(packed);
    TORCH_CHECK(packed.is_contiguous() && escapes.is_contiguous(),
                "nf12_hip.linear: packed and escapes must be contiguous");
    TORCH_CHECK(exp_base >= 0 && exp_base <= kMaxExpBase,
                "nf12_hip.linear: exp_base must be in [0, 120], got ", exp_base);
    const int64_t row_bytes = packed.size(1);
    TORCH_CHECK(n > 0 && n % 16 == 0, "nf12_hip.linear: N must be a positive multiple of 16");
    TORCH_CHECK(row_bytes % 96 == 0,
                "nf12_hip.linear: packed row must be a multiple of 96 bytes");
    TORCH_CHECK(k % 64 == 0, "nf12_hip.linear: K must be a multiple of 64");
    TORCH_CHECK(escapes.scalar_type() == at::kShort && escapes.dim() == 2 &&
                    escapes.size(1) == 8,
                "nf12_hip.linear: escapes must be int16 [E, 8]");
    auto out = torch::empty({m, n}, x.options());
    // Bias is folded into the kernel's FP32 accumulator, not added to the
    // rounded BF16 result (see the comment in nf12_decode.hip).
    const bool has_bias = bias.has_value() && bias->defined();
    const torch::Tensor b =
        has_bias ? bias->to(out.device(), out.scalar_type()).contiguous() : torch::Tensor();
    if (has_bias) {
      TORCH_CHECK(b.numel() == n, "nf12_hip.linear: bias must have N elements");
    }
    const hipError_t status = nf12_linear_fused(
        packed.data_ptr<uint8_t>(), reinterpret_cast<const uint4*>(escapes.data_ptr<int16_t>()),
        static_cast<uint32_t>(exp_base), static_cast<const void*>(x2.data_ptr()),
        static_cast<void*>(out.data_ptr()), n, k, m,
        has_bias ? static_cast<const void*>(b.data_ptr()) : nullptr, current_stream());
    TORCH_CHECK(status == hipSuccess, "nf12_hip.linear (fused) failed: ",
                hipGetErrorString(status));
    return out.reshape(out_sizes);
  }

  // Larger M: decode the weight and hand it to rocBLAS (SPEC.md 6.5). Decode
  // row slices sized to the L2 cache rather than the whole matrix, so the
  // decoded tile is still hot when the GEMM reads it. A full-matrix decode
  // spills 32 MiB+ straight past an 8 MiB RDNA4 L2 and the GEMM re-reads it
  // from memory.
  const int64_t row_bytes = packed.size(1);
  TORCH_CHECK(row_bytes * 2 / 3 == k, "nf12_hip.linear: packed K mismatch");
  TORCH_CHECK(n % 16 == 0 && row_bytes % 96 == 0 && k % 64 == 0,
              "nf12_hip.linear: unsupported NF12 shape");

  // Mid-M: matrix cores. Decode-then-GEMM materialises the dense weight, so it
  // moves 88 MiB for a 4096x4096 matrix where a BF16 GEMM moves 32 MiB and the
  // packed form is 24 MiB. This path decodes each super-atom once into shared
  // memory and feeds wmma for up to eight token tiles, so the packed bytes are
  // read once and the dense weight never exists. It is selected before the
  // rocBLAS route and skipped on gfx1250, which has no wmma-128b-insts.
  if (m > 0 && nf12_wmma_supported(m)) {
    TORCH_CHECK(n > 0 && n % 16 == 0, "nf12_hip.linear: N must be a positive multiple of 16");
    TORCH_CHECK(packed.is_contiguous() && escapes.is_contiguous(),
                "nf12_hip.linear: packed and escapes must be contiguous");
    TORCH_CHECK(exp_base >= 0 && exp_base <= kMaxExpBase,
                "nf12_hip.linear: exp_base must be in [0, 120], got ", exp_base);
    TORCH_CHECK(row_bytes % 96 == 0 && k % 64 == 0,
                "nf12_hip.linear: unsupported NF12 shape");
    const void* bias_ptr = nullptr;
    std::optional<torch::Tensor> b_opt;
    if (bias.has_value() && bias->defined()) {
      TORCH_CHECK(bias->numel() == n, "nf12_hip.linear: bias must have N elements");
      b_opt = bias->to(x.device(), x.scalar_type()).contiguous();
      bias_ptr = b_opt->data_ptr();
    }
    auto out = torch::empty({m, n}, x.options());
    const size_t workspace_bytes = nf12_wmma_workspace_bytes(n, k, m);
    // PyTorch's caching allocator owns this workspace for the call and tracks
    // the current stream. Concurrent streams and devices cannot alias it.
    torch::Tensor workspace;
    if (workspace_bytes > 0) {
      workspace = torch::empty({static_cast<int64_t>(workspace_bytes)}, x.options().dtype(at::kByte));
    }
    const hipError_t status = nf12_linear_wmma(
        packed.data_ptr<uint8_t>(),
        reinterpret_cast<const uint4*>(escapes.data_ptr<int16_t>()),
        static_cast<uint32_t>(exp_base), static_cast<const void*>(x2.data_ptr()),
        static_cast<void*>(out.data_ptr()), n, k, m, bias_ptr,
        workspace_bytes > 0 ? workspace.data_ptr() : nullptr, workspace_bytes,
        current_stream());
    TORCH_CHECK(status == hipSuccess, "nf12_hip.linear (wmma) failed: ",
                hipGetErrorString(status));
    return out.reshape(out_sizes);
  }

  // Slice height: large enough that rocBLAS keeps its efficiency (a 256-row
  // GEMM is launch-bound), small enough that the decoded tile still lands in
  // L2. 1024 rows measured best on gfx1201 across M = 16..256; the L2 term
  // only ever raises it. Below ~1024 rows the extra GEMM launches cost more
  // than the cache locality buys.
  constexpr int64_t kMinSliceRows = 1024;
  int64_t slice_rows = n;
  const int64_t l2_bytes = static_cast<int64_t>(at::cuda::getCurrentDeviceProperties()->l2CacheSize);
  if (l2_bytes > 0 && n > kMinSliceRows) {
    const int64_t budget = l2_bytes / 2;  // decoded BF16 tile only
    const int64_t rows = std::max(kMinSliceRows, (budget / (k * 2)) / 16 * 16);
    slice_rows = std::min(n, rows);
  }

  // Collect per-slice results and concatenate once. Writing each slice into a
  // column view of a preallocated output looks cheaper but the view is
  // non-contiguous, so the copy_ becomes a strided element-wise pass that cost
  // far more than the GEMM it was meant to accelerate.
  std::vector<torch::Tensor> slice_out;
  slice_out.reserve((n + slice_rows - 1) / slice_rows);
  for (int64_t n0 = 0; n0 < n; n0 += slice_rows) {
    const int64_t rows = std::min(slice_rows, n - n0);
    // A row slice of an NF12 weight is the packed view [n0, n0+rows); escape
    // indices are absolute, so the full table applies (SPEC.md 6.5).
    const torch::Tensor ps = packed.narrow(0, n0, rows);
    const torch::Tensor ws = decode_dispatch(ps, escapes, exp_base);
    if (bias.has_value() && bias->defined()) {
      // Fold the bias into the GEMM epilogue (addmm), never add it to an
      // already-rounded BF16 product. Adding afterwards double-rounds: the
      // matmul rounds H, then H + bias rounds again, which measured 0.0017
      // RMS relative error and only 80% bit-exact outputs on gfx1201 versus
      // 2e-5 / 99.99% for addmm. The CUDA kernel also accumulates bias in
      // FP32 before writing BF16 (csrc/linear/small_m.cu), so addmm matches it.
      const torch::Tensor bs = bias->to(ws.device(), ws.scalar_type()).narrow(0, n0, rows);
      slice_out.push_back(at::addmm(bs, x2, ws.t()));
    } else {
      slice_out.push_back(at::matmul(x2, ws.t()));
    }
  }
  const torch::Tensor dense = slice_out.size() == 1 ? slice_out[0] : at::cat(slice_out, 1);
  return dense.reshape(out_sizes);
}

// ---------------------------------------------------------------------------
// nf12_hip::selftest(ref) -> bool
// ---------------------------------------------------------------------------
bool selftest(const torch::Tensor& ref) {
  check_arch(ref);
  auto guard = device_guard(ref);
  auto sentinel = torch::zeros({1}, ref.options().dtype(at::kInt));
  const hipError_t status = nf12_selftest(
      reinterpret_cast<uint32_t*>(sentinel.data_ptr<int32_t>()), current_stream());
  TORCH_CHECK(status == hipSuccess, "nf12_hip.selftest failed: ", hipGetErrorString(status));
  return sentinel.cpu().data_ptr<int32_t>()[0] == 0x4E463132;
}

// ---------------------------------------------------------------------------
// nf12_hip::arch_info() -> str
// ---------------------------------------------------------------------------
std::string arch_info() {
  TORCH_CHECK(torch::cuda::is_available(), "nf12_hip.arch_info: no GPU");
  return arch_name(c10::cuda::current_device());
}

}  // namespace

TORCH_LIBRARY(nf12_hip, m) {
  m.def("decode(Tensor packed, Tensor escapes, int exp_base) -> Tensor");
  m.def("decode_fp32(Tensor packed, Tensor escapes, int exp_base) -> Tensor");
  m.def("sub_fp32(Tensor packed, Tensor escapes, int exp_base, Tensor wq) -> Tensor");
  m.def("embedding(Tensor ids, Tensor packed, Tensor escapes, int exp_base) -> Tensor");
  m.def("rowmajor_pack(Tensor packed, Tensor escapes, int exp_base) -> "
        "(Tensor row_packed, Tensor row_escapes, int original_escapes, int collisions)");
  m.def("rowmajor_embedding(Tensor ids, Tensor row_packed, Tensor row_escapes, int exp_base) -> Tensor");
  m.def(
      "linear(Tensor x, Tensor packed, Tensor escapes, int exp_base, "
      "Tensor? bias=None) -> Tensor");
  m.def("encode(Tensor w) -> (Tensor packed, Tensor escapes, int exp_base)");
  m.def("selftest(Tensor ref) -> bool");
  m.def("arch_info() -> str");
}

TORCH_LIBRARY_IMPL(nf12_hip, CUDA, m) {
  m.impl("decode", &decode_dispatch);
  m.impl("decode_fp32", &decode_fp32_dispatch);
  m.impl("sub_fp32", &sub_fp32_dispatch);
  m.impl("embedding", &embedding_dispatch);
  m.impl("rowmajor_pack", &rowmajor_pack_dispatch);
  m.impl("rowmajor_embedding", &rowmajor_embedding_dispatch);
  m.impl("linear", &linear_dispatch);
  m.impl("encode", &encode_dispatch);
  m.impl("selftest", &selftest);
}

// arch_info() takes no tensor arguments, so it cannot dispatch on a key set
// derived from inputs. CompositeExplicitAutograd makes it callable directly.
TORCH_LIBRARY_IMPL(nf12_hip, CompositeExplicitAutograd, m) {
  m.impl("arch_info", &arch_info);
}
