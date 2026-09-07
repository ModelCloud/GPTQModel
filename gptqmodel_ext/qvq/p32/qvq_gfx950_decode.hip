// SPDX-License-Identifier: Apache-2.0
// Experimental native window decoder. Not selected by the production ABI.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstring>

namespace {
// One wave owns one tile: lane and lane+64 share extraction shift. This
// storage-neutral pairing comes from QVQ's Hopper/Ampere window decoders.
// Four waves per CTA; no cross-wave communication or implicit barriers.
template <unsigned Bits>
__global__ void decode_window(const uint32_t* window, const __half* levels,
                              const uint8_t* banks, __half* dense,
                              unsigned k, unsigned n, uint32_t alternate) {
  const unsigned lane = threadIdx.x & 63;
  const unsigned nt = n / 16;
  const unsigned ntile = blockIdx.x * 4 + (threadIdx.x >> 6);
  if (ntile >= nt) return;
  const unsigned kt = blockIdx.y;
  const unsigned nc = ntile * 16;
  const uint64_t tile = uint64_t(kt) * nt + ntile;
  const uint32_t* words = window + tile * (4 * Bits);
  const unsigned bit = (127 - lane) * Bits;
  const unsigned word = bit >> 5;
  const unsigned shift = bit & 31;
  const unsigned next = word + 1 == 4 * Bits ? 0 : word + 1;
  const uint32_t first = __builtin_amdgcn_alignbit(words[next], words[word], shift);
  const uint32_t second = __builtin_amdgcn_alignbit(
      words[word - 2 * Bits + 1], words[word - 2 * Bits], shift);
  const uint32_t bank = banks[tile];
  #pragma unroll
  for (unsigned half = 0; half < 2; ++half) {
    const unsigned pair = lane + half * 64;
    uint32_t state = (half ? second : first) & 0xffff;
    state ^= ((bank >> (pair >> 4)) & 1) * alternate;
    state ^= state >> 8;
    state = (state * 40503u + 17011u) & 0xffff;
    state ^= state >> 7;
    const unsigned col = nc + (pair & 7) * 2;
    const unsigned row = kt * 16 + (pair >> 3);
    dense[uint64_t(col) * k + row] = levels[state >> 8];
    dense[uint64_t(col + 1) * k + row] = levels[state & 255];
  }
}
}  // namespace

// Caller owns all buffers through completion/replay. No allocation, hidden
// synchronization, cache or stream mutation. Dense is contiguous FP16 [N,K].
// Dimensions and pointers are a trusted device-buffer contract, as in the ABI.
extern "C" int qvq_gfx950_decode_window(const void* window, const void* levels,
    const void* banks, void* dense, unsigned k, unsigned n, unsigned bits,
    unsigned bank_alt_id, void* stream) {
  if (!window || !levels || !banks || !dense || !k || !n || k % 16 || n % 16 ||
      bits < 4 || bits > 7 || bank_alt_id > 3) return hipErrorInvalidValue;
  int device;
  hipDeviceProp_t props{};
  hipError_t status = hipGetDevice(&device);
  if (status != hipSuccess) return status;
  status = hipGetDeviceProperties(&props, device);
  if (status != hipSuccess) return status;
  if (std::strncmp(props.gcnArchName, "gfx950", 6) != 0 ||
      (props.gcnArchName[6] != '\0' && props.gcnArchName[6] != ':'))
    return hipErrorNoBinaryForGpu;
  constexpr uint32_t masks[4][4] = {
      {0, 0x5a5a, 0x3c3c, 0xc3c3}, {0, 0x9696, 0x3c3c, 0xc3c3},
      {0, 0x6969, 0x5a5a, 0x3c3c}, {0, 0xc3c3, 0x9696, 0x5a5a}};
  // Two-dimensional ownership eliminates runtime division/remainder in every
  // wave. Limit Y to the portable HIP grid bound; do not truncate dimensions.
  if (k / 16 > 65535) return hipErrorInvalidValue;
  const unsigned blocks = (n / 16 + 3) / 4;
  const auto launch = [&]<unsigned B>() {
    hipLaunchKernelGGL(decode_window<B>, dim3(blocks, k / 16), dim3(256), 0,
        static_cast<hipStream_t>(stream), static_cast<const uint32_t*>(window),
        static_cast<const __half*>(levels), static_cast<const uint8_t*>(banks),
        static_cast<__half*>(dense), k, n, masks[B - 4][bank_alt_id]);
  };
  switch (bits) {
    case 4: launch.template operator()<4>(); break;
    case 5: launch.template operator()<5>(); break;
    case 6: launch.template operator()<6>(); break;
    case 7: launch.template operator()<7>(); break;
  }
  return hipGetLastError();
}
