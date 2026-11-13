// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0
// ---------------------------------------------------------------------------
// Standalone hipcc self-test for the Tenstorrent block-float kernels.
//
// This exists so the kernels can be validated on a ROCm host that has no
// ROCm-built PyTorch (the JIT extension is still the production entry point).
// It exercises dequant, quantize, decompose and the fused linear kernels
// against an independent CPU implementation, prints throughput numbers, and
// can dump the exact inputs/outputs for a Torch cross-check:
//
//   hipcc --offload-arch=gfx1201 -O3 -std=c++17 \
//       tt_blockfloat_hip.hip tt_blockfloat_hip_selftest.cpp -o ttbf_selftest
//   ./ttbf_selftest            # internal checks + timings
//   ./ttbf_selftest --dump DIR # also write DIR/*.bin for verify_against_torch.py
// ---------------------------------------------------------------------------

#include "tt_blockfloat_hip.h"

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int kLevels[2] = {4, 64};  // indexed by bits==4 ? 0 : 1
constexpr int kOffset[2] = {7, 127};

int bits_index(int bits) { return bits == 4 ? 0 : 1; }

float ref_value(int code, int bits, int exponent) {
    const int idx = bits_index(bits);
    const float scale = std::exp2(static_cast<float>(exponent));
    return static_cast<float>(code - kOffset[idx]) /
           static_cast<float>(kLevels[idx]) * scale;
}

void ref_quantise_block(const float* x, int bits, int* code_out, int* exp_out) {
    const int idx = bits_index(bits);
    const int levels = kLevels[idx];
    const int offset = kOffset[idx];
    float magnitude = 0.0f;
    for (int i = 0; i < 16; ++i) {
        magnitude = std::max(magnitude, std::fabs(x[i]));
    }
    const float safe = std::max(magnitude, 1.17549435e-38f);
    const int base = static_cast<int>(std::floor(std::log2(safe)));

    float best_error = INFINITY;
    int best_exp = base;
    int best_code[16];
    for (int candidate = 0; candidate < 3; ++candidate) {
        const int exponent = base + (candidate == 0 ? 0 : (candidate == 1 ? -1 : 1));
        const float scale = std::exp2(static_cast<float>(exponent));
        float error = 0.0f;
        int code[16];
        for (int i = 0; i < 16; ++i) {
            float q = std::rint((x[i] / scale) * static_cast<float>(levels));
            q = std::min(std::max(q, -static_cast<float>(offset)),
                         static_cast<float>(offset));
            code[i] = static_cast<int>(q);
            const float reconstructed =
                (static_cast<float>(code[i]) / static_cast<float>(levels)) * scale;
            const float diff = reconstructed - x[i];
            error += diff * diff;
        }
        // Candidate 0 is taken unconditionally, matching the Torch reference.
        // Comparing strictly against INFINITY leaves ``best_code`` untouched
        // when every candidate overflows, which happens for large inputs.
        if (candidate == 0 || error < best_error) {
            best_error = error;
            best_exp = exponent;
            std::memcpy(best_code, code, sizeof(code));
        }
    }
    *exp_out = best_exp;
    std::memcpy(code_out, best_code, sizeof(best_code));
}

void ref_quantize(const std::vector<float>& x, int bits, int64_t rows,
                  int64_t k, std::vector<uint8_t>* codes,
                  std::vector<int8_t>* exponents) {
    const int64_t blocks_per_row = k / 16;
    const int64_t code_bytes =
        bits == 4 ? rows * k / 2 : rows * k;
    codes->assign(code_bytes, 0);
    exponents->assign(rows * blocks_per_row, 0);
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t b = 0; b < blocks_per_row; ++b) {
            int code[16];
            int exponent = 0;
            ref_quantise_block(x.data() + r * k + b * 16, bits, code, &exponent);
            (*exponents)[r * blocks_per_row + b] =
                static_cast<int8_t>(exponent);
            uint8_t* dst = codes->data() + (r * k + b * 16) / (bits == 4 ? 2 : 1);
            if (bits == 4) {
                for (int j = 0; j < 8; ++j) {
                    const uint8_t lo =
                        static_cast<uint8_t>(code[2 * j] + kOffset[0]);
                    const uint8_t hi =
                        static_cast<uint8_t>(code[2 * j + 1] + kOffset[0]);
                    dst[j] = static_cast<uint8_t>(lo | (hi << 4));
                }
            } else {
                for (int j = 0; j < 16; ++j) {
                    dst[j] = static_cast<uint8_t>(code[j] + kOffset[1]);
                }
            }
        }
    }
}

void ref_dequant(const std::vector<uint8_t>& codes,
                 const std::vector<int8_t>& exponents, int bits, int64_t rows,
                 int64_t k, std::vector<float>* out) {
    out->assign(rows * k, 0.0f);
    const int64_t blocks_per_row = k / 16;
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < k; ++c) {
            const int exponent =
                exponents[r * blocks_per_row + c / 16];
            int code = 0;
            if (bits == 4) {
                const uint8_t byte =
                    codes[(r * k + c) / 2];
                code = (c & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
            } else {
                code = codes[r * k + c];
            }
            (*out)[r * k + c] = ref_value(code, bits, exponent);
        }
    }
}

void ref_linear(const std::vector<float>& x,
                const std::vector<uint8_t>& codes,
                const std::vector<int8_t>& exponents, int bits, int64_t m,
                int64_t n, int64_t k, const std::vector<float>* bias,
                std::vector<float>* out) {
    std::vector<float> weight;
    ref_dequant(codes, exponents, bits, n, k, &weight);
    out->assign(m * n, 0.0f);
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            double acc = 0.0;
            for (int64_t c = 0; c < k; ++c) {
                acc += static_cast<double>(x[i * k + c]) *
                       static_cast<double>(weight[j * k + c]);
            }
            float value = static_cast<float>(acc);
            if (bias != nullptr) {
                value += (*bias)[j];
            }
            (*out)[i * n + j] = value;
        }
    }
}

template <typename T>
float to_float(T v);

template <>
float to_float<float>(float v) {
    return v;
}

template <>
float to_float<__half>(__half v) {
    return __half2float(v);
}

template <>
float to_float<__hip_bfloat16>(__hip_bfloat16 v) {
    return __bfloat162float(v);
}

template <typename T>
T from_float(float v);

template <>
float from_float<float>(float v) {
    return v;
}

template <>
__half from_float<__half>(float v) {
    return __float2half_rn(v);
}

template <>
__hip_bfloat16 from_float<__hip_bfloat16>(float v) {
    return __float2bfloat16(v);
}

int failures = 0;

void report(bool ok, const char* what, const std::string& detail = "") {
    if (!ok) {
        ++failures;
    }
    std::printf("%-52s %s%s%s\n", what, ok ? "PASS" : "FAIL",
                detail.empty() ? "" : "  ", detail.c_str());
}

template <typename T>
bool device_dequant(const std::vector<uint8_t>& codes,
                    const std::vector<int8_t>& exponents, int bits, int64_t rows,
                    int64_t k, int dtype_code, std::vector<float>* out) {
    uint8_t* d_codes = nullptr;
    int8_t* d_exp = nullptr;
    T* d_out = nullptr;
    if (hipMalloc(&d_codes, codes.size()) != hipSuccess ||
        hipMalloc(&d_exp, exponents.size()) != hipSuccess ||
        hipMalloc(&d_out, rows * k * sizeof(T)) != hipSuccess) {
        return false;
    }
    hipMemcpy(d_codes, codes.data(), codes.size(), hipMemcpyHostToDevice);
    hipMemcpy(d_exp, exponents.data(), exponents.size(), hipMemcpyHostToDevice);
    const hipError_t status = ttbf_dequant(d_codes, d_exp, bits, rows, k, d_out,
                                           dtype_code, nullptr);
    if (status != hipSuccess) {
        std::printf("ttbf_dequant: %s\n", hipGetErrorString(status));
        return false;
    }
    std::vector<T> host(rows * k);
    hipMemcpy(host.data(), d_out, rows * k * sizeof(T), hipMemcpyDeviceToHost);
    out->resize(rows * k);
    for (size_t i = 0; i < host.size(); ++i) {
        (*out)[i] = to_float<T>(host[i]);
    }
    hipFree(d_codes);
    hipFree(d_exp);
    hipFree(d_out);
    return true;
}

void write_bin(const std::string& path, const void* data, size_t bytes) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) {
        std::printf("could not open %s\n", path.c_str());
        return;
    }
    std::fwrite(data, 1, bytes, f);
    std::fclose(f);
}

}  // namespace

int main(int argc, char** argv) {
    std::string dump_dir;
    int64_t bench_m = 32;
    int64_t bench_n = 4096;
    int64_t bench_k = 4096;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--dump") == 0 && i + 1 < argc) {
            dump_dir = argv[++i];
        } else if (std::strcmp(argv[i], "--bench") == 0 && i + 3 < argc) {
            bench_m = std::atoll(argv[++i]);
            bench_n = std::atoll(argv[++i]);
            bench_k = std::atoll(argv[++i]);
        }
    }
    if (!dump_dir.empty()) {
        std::filesystem::create_directories(dump_dir);
    }

    hipDeviceProp_t props{};
    hipGetDeviceProperties(&props, 0);
    std::printf("device: %s (%s), %d CUs\n", props.name, props.gcnArchName,
                props.multiProcessorCount);

    int32_t* d_sentinel = nullptr;
    hipMalloc(&d_sentinel, sizeof(int32_t));
    const hipError_t sentinel_status = ttbf_selftest(d_sentinel, nullptr);
    int32_t sentinel = 0;
    hipMemcpy(&sentinel, d_sentinel, sizeof(sentinel), hipMemcpyDeviceToHost);
    report(sentinel_status == hipSuccess && sentinel == 0x54544642,
           "selftest kernel executed");
    hipFree(d_sentinel);

    std::mt19937 rng(12345);
    const int64_t rows = 128;
    const int64_t k = 128;

    for (int bits : {4, 8}) {
        const int idx = bits_index(bits);
        const int code_max = 2 * kOffset[idx];

        // --- dequant ---------------------------------------------------------
        std::vector<uint8_t> codes(bits == 4 ? rows * k / 2 : rows * k);
        for (auto& byte : codes) {
            byte = static_cast<uint8_t>(rng() % (code_max + 1));
        }
        std::vector<int8_t> exponents(rows * (k / 16));
        for (auto& e : exponents) {
            e = static_cast<int8_t>(static_cast<int>(rng() % 41) - 20);
        }
        std::vector<float> expected;
        ref_dequant(codes, exponents, bits, rows, k, &expected);

        std::vector<float> got_f32;
        bool ok = device_dequant<float>(codes, exponents, bits, rows, k,
                                        TTBF_DTYPE_F32, &got_f32);
        report(ok && got_f32 == expected,
               bits == 4 ? "bf4 dequant f32 bit-exact"
                         : "bf8 dequant f32 bit-exact");

        std::vector<float> got_bf16;
        ok = device_dequant<__hip_bfloat16>(codes, exponents, bits, rows, k,
                                            TTBF_DTYPE_BF16, &got_bf16);
        bool bf16_ok = ok;
        for (size_t i = 0; i < expected.size() && bf16_ok; ++i) {
            bf16_ok = got_bf16[i] ==
                      __bfloat162float(__float2bfloat16(expected[i]));
        }
        report(bf16_ok, bits == 4 ? "bf4 dequant bf16 rounds correctly"
                                  : "bf8 dequant bf16 rounds correctly");

        // --- quantize --------------------------------------------------------
        std::vector<float> input(rows * k);
        std::normal_distribution<float> normal(0.0f, 1.0f);
        for (int64_t r = 0; r < rows; ++r) {
            const float block_scale =
                std::exp2(static_cast<float>(static_cast<int>(rng() % 41) - 20));
            for (int64_t c = 0; c < k; ++c) {
                input[r * k + c] = normal(rng) * block_scale;
            }
        }
        // A row that straddles binade boundaries, including exact powers of two.
        for (int64_t c = 0; c < k; ++c) {
            const float p = std::exp2(static_cast<float>((c % 17) - 8));
            input[c] = (c % 3 == 0) ? std::nextafter(p, 0.0f) : p;
        }

        std::vector<uint8_t> ref_codes;
        std::vector<int8_t> ref_exponents;
        ref_quantize(input, bits, rows, k, &ref_codes, &ref_exponents);

        uint8_t* d_codes = nullptr;
        int8_t* d_exp = nullptr;
        float* d_x = nullptr;
        hipMalloc(&d_codes, ref_codes.size());
        hipMalloc(&d_exp, ref_exponents.size());
        hipMalloc(&d_x, input.size() * sizeof(float));
        hipMemcpy(d_x, input.data(), input.size() * sizeof(float),
                  hipMemcpyHostToDevice);
        hipError_t status =
            ttbf_quantize(d_x, bits, rows, k, d_codes, d_exp, nullptr);
        std::vector<uint8_t> got_codes(ref_codes.size());
        std::vector<int8_t> got_exponents(ref_exponents.size());
        hipMemcpy(got_codes.data(), d_codes, got_codes.size(),
                  hipMemcpyDeviceToHost);
        hipMemcpy(got_exponents.data(), d_exp, got_exponents.size(),
                  hipMemcpyDeviceToHost);
        report(status == hipSuccess && got_codes == ref_codes &&
                   got_exponents == ref_exponents,
               bits == 4 ? "bf4 quantize matches CPU oracle"
                         : "bf8 quantize matches CPU oracle");

        // --- quantize with overflowing squared error -------------------------
        // Near FLT_MAX every candidate's squared error is inf, so a strict
        // ``error < best_error`` never initialises the winning codes.  Compare
        // against the reference *and* assert the magnitude: both sides could
        // regress to "uninitialised" together and still compare equal.
        {
            const int64_t kk = 16;
            std::vector<float> huge(kk, 1e30f);
            std::vector<uint8_t> hcodes(bits == 4 ? kk / 2 : kk);
            std::vector<int8_t> hexp(1);
            ref_quantize(huge, bits, 1, kk, &hcodes, &hexp);

            uint8_t* dh_codes = nullptr;
            int8_t* dh_exp = nullptr;
            float* dh_x = nullptr;
            hipMalloc(&dh_codes, hcodes.size());
            hipMalloc(&dh_exp, hexp.size());
            hipMalloc(&dh_x, huge.size() * sizeof(float));
            hipMemcpy(dh_x, huge.data(), huge.size() * sizeof(float),
                      hipMemcpyHostToDevice);
            const hipError_t hst =
                ttbf_quantize(dh_x, bits, 1, kk, dh_codes, dh_exp, nullptr);
            std::vector<uint8_t> hgot(hcodes.size());
            std::vector<int8_t> hgot_exp(hexp.size());
            hipMemcpy(hgot.data(), dh_codes, hgot.size(),
                      hipMemcpyDeviceToHost);
            hipMemcpy(hgot_exp.data(), dh_exp, hgot_exp.size(),
                      hipMemcpyDeviceToHost);

            std::vector<float> hdec;
            ref_dequant(hgot, hgot_exp, bits, 1, kk, &hdec);
            const float reconstructed = hdec.empty() ? 0.0f : hdec[0];
            report(hst == hipSuccess && hgot == hcodes && hgot_exp == hexp &&
                       reconstructed > 5e29f,
                   bits == 4 ? "bf4 quantize survives overflow"
                             : "bf8 quantize survives overflow");
            hipFree(dh_codes);
            hipFree(dh_exp);
            hipFree(dh_x);
        }

        if (!dump_dir.empty()) {
            const std::string suffix = bits == 4 ? "4" : "8";
            // Dump dequantisation of the *quantised* codes, so the Torch
            // verifier can compare against quantize_blockfloat's own output.
            std::vector<float> quant_f32;
            std::vector<float> quant_bf16;
            device_dequant<float>(got_codes, got_exponents, bits, rows, k,
                                  TTBF_DTYPE_F32, &quant_f32);
            device_dequant<__hip_bfloat16>(got_codes, got_exponents, bits, rows,
                                           k, TTBF_DTYPE_BF16, &quant_bf16);
            write_bin(dump_dir + "/q" + suffix + "_in.bin", input.data(),
                      input.size() * sizeof(float));
            write_bin(dump_dir + "/q" + suffix + "_codes.bin", got_codes.data(),
                      got_codes.size());
            write_bin(dump_dir + "/q" + suffix + "_exp.bin",
                      got_exponents.data(), got_exponents.size());
            write_bin(dump_dir + "/d" + suffix + "_out_f32.bin",
                      quant_f32.data(), quant_f32.size() * sizeof(float));
            write_bin(dump_dir + "/d" + suffix + "_out_bf16_as_f32.bin",
                      quant_bf16.data(), quant_bf16.size() * sizeof(float));
        }

        // --- decompose -------------------------------------------------------
        std::vector<float> grid;
        ref_dequant(got_codes, got_exponents, bits, rows, k, &grid);
        float* d_grid = nullptr;
        uint8_t* d_grid_codes = nullptr;
        int8_t* d_grid_exp = nullptr;
        int32_t* d_failed = nullptr;
        hipMalloc(&d_grid, grid.size() * sizeof(float));
        hipMalloc(&d_grid_codes, got_codes.size());
        hipMalloc(&d_grid_exp, got_exponents.size());
        hipMalloc(&d_failed, sizeof(int32_t));
        hipMemcpy(d_grid, grid.data(), grid.size() * sizeof(float),
                  hipMemcpyHostToDevice);
        hipMemset(d_failed, 0, sizeof(int32_t));
        status = ttbf_decompose(d_grid, bits, rows, k, d_grid_codes, d_grid_exp,
                                d_failed, nullptr);
        std::vector<uint8_t> dec_codes(got_codes.size());
        std::vector<int8_t> dec_exp(got_exponents.size());
        int32_t failed = 0;
        hipMemcpy(dec_codes.data(), d_grid_codes, dec_codes.size(),
                  hipMemcpyDeviceToHost);
        hipMemcpy(dec_exp.data(), d_grid_exp, dec_exp.size(),
                  hipMemcpyDeviceToHost);
        hipMemcpy(&failed, d_failed, sizeof(failed), hipMemcpyDeviceToHost);
        std::vector<float> roundtrip;
        ref_dequant(dec_codes, dec_exp, bits, rows, k, &roundtrip);
        report(status == hipSuccess && failed == 0 && roundtrip == grid,
               bits == 4 ? "bf4 decompose round-trips exactly"
                         : "bf8 decompose round-trips exactly");

        // --- fused linear ----------------------------------------------------
        const int64_t n = 128;
        std::vector<float> bias(n);
        for (int64_t j = 0; j < n; ++j) {
            bias[j] = 0.01f * static_cast<float>(static_cast<int>(rng() % 200) - 100);
        }
        for (int64_t m : {static_cast<int64_t>(1), static_cast<int64_t>(32)}) {
            std::vector<float> act(m * k);
            for (auto& v : act) {
                v = normal(rng);
            }
            std::vector<float> ref;
            ref_linear(act, got_codes, got_exponents, bits, m, n, k, &bias,
                       &ref);

            float* d_act = nullptr;
            float* d_bias = nullptr;
            float* d_out = nullptr;
            hipMalloc(&d_act, act.size() * sizeof(float));
            hipMalloc(&d_bias, bias.size() * sizeof(float));
            hipMalloc(&d_out, ref.size() * sizeof(float));
            hipMemcpy(d_act, act.data(), act.size() * sizeof(float),
                      hipMemcpyHostToDevice);
            hipMemcpy(d_bias, bias.data(), bias.size() * sizeof(float),
                      hipMemcpyHostToDevice);
            status = ttbf_linear(d_act, TTBF_DTYPE_F32, d_codes, d_exp, bits, m,
                                 n, k, d_bias, d_out, nullptr);
            std::vector<float> out(ref.size());
            hipMemcpy(out.data(), d_out, out.size() * sizeof(float),
                      hipMemcpyDeviceToHost);

            double abs_error = 0.0;
            double ref_mass = 0.0;
            for (size_t i = 0; i < ref.size(); ++i) {
                abs_error += std::fabs(static_cast<double>(out[i] - ref[i]));
                ref_mass += std::fabs(static_cast<double>(ref[i]));
            }
            const double norm_l1 = abs_error / std::max(1e-6, ref_mass);
            char label[96];
            std::snprintf(label, sizeof(label),
                          "%s linear m=%lld norm_l1=%.3g",
                          bits == 4 ? "bf4" : "bf8",
                          static_cast<long long>(m), norm_l1);
            report(status == hipSuccess && norm_l1 < 2e-3, label);

            if (!dump_dir.empty() && bits == 4 && m == 32) {
                write_bin(dump_dir + "/lin_x.bin", act.data(),
                          act.size() * sizeof(float));
                write_bin(dump_dir + "/lin_bias.bin", bias.data(),
                          bias.size() * sizeof(float));
                write_bin(dump_dir + "/lin_codes.bin", got_codes.data(),
                          got_codes.size());
                write_bin(dump_dir + "/lin_exp.bin", got_exponents.data(),
                          got_exponents.size());
                write_bin(dump_dir + "/lin_out.bin", out.data(),
                          out.size() * sizeof(float));
            }
            hipFree(d_act);
            hipFree(d_bias);
            hipFree(d_out);
        }

        // --- bf16 prefill (matrix-core path on gfx12) ------------------------
        {
            const int64_t m = 32;
            std::vector<float> act(m * k);
            std::vector<float> bias_f(n);
            for (auto& v : act) {
                v = normal(rng);
            }
            for (auto& v : bias_f) {
                v = 0.01f * static_cast<float>(static_cast<int>(rng() % 200) - 100);
            }
            std::vector<__hip_bfloat16> act_b(m * k);
            std::vector<__hip_bfloat16> bias_b(n);
            std::vector<__hip_bfloat16> out_b(m * n);
            std::vector<float> act_rounded(m * k);
            std::vector<float> bias_rounded(n);
            for (size_t i = 0; i < act.size(); ++i) {
                act_b[i] = from_float<__hip_bfloat16>(act[i]);
                act_rounded[i] = to_float<__hip_bfloat16>(act_b[i]);
            }
            for (size_t i = 0; i < bias_f.size(); ++i) {
                bias_b[i] = from_float<__hip_bfloat16>(bias_f[i]);
                bias_rounded[i] = to_float<__hip_bfloat16>(bias_b[i]);
            }
            std::vector<float> ref;
            ref_linear(act_rounded, got_codes, got_exponents, bits, m, n, k,
                       &bias_rounded, &ref);
            __hip_bfloat16* d_act_b = nullptr;
            __hip_bfloat16* d_bias_b = nullptr;
            __hip_bfloat16* d_out_b = nullptr;
            hipMalloc(&d_act_b, act_b.size() * sizeof(__hip_bfloat16));
            hipMalloc(&d_bias_b, bias_b.size() * sizeof(__hip_bfloat16));
            hipMalloc(&d_out_b, out_b.size() * sizeof(__hip_bfloat16));
            hipMemcpy(d_act_b, act_b.data(),
                      act_b.size() * sizeof(__hip_bfloat16),
                      hipMemcpyHostToDevice);
            hipMemcpy(d_bias_b, bias_b.data(),
                      bias_b.size() * sizeof(__hip_bfloat16),
                      hipMemcpyHostToDevice);
            status = ttbf_linear(d_act_b, TTBF_DTYPE_BF16, d_codes, d_exp, bits,
                                 m, n, k, d_bias_b, d_out_b, nullptr);
            hipMemcpy(out_b.data(), d_out_b,
                      out_b.size() * sizeof(__hip_bfloat16),
                      hipMemcpyDeviceToHost);

            double abs_error = 0.0;
            double ref_mass = 0.0;
            for (size_t i = 0; i < out_b.size(); ++i) {
                const float rounded = to_float<__hip_bfloat16>(
                    from_float<__hip_bfloat16>(ref[i]));
                abs_error += std::fabs(static_cast<double>(
                    to_float<__hip_bfloat16>(out_b[i]) - rounded));
                ref_mass += std::fabs(static_cast<double>(rounded));
            }
            const double norm_l1 = abs_error / std::max(1e-6, ref_mass);
            char label[96];
            std::snprintf(label, sizeof(label),
                          "%s linear bf16 m=32 norm_l1=%.3g",
                          bits == 4 ? "bf4" : "bf8", norm_l1);
            report(status == hipSuccess && norm_l1 < 2e-3, label);
            if (!dump_dir.empty() && bits == 4) {
                write_bin(dump_dir + "/lin_bf16_x.bin", act_b.data(),
                          act_b.size() * sizeof(__hip_bfloat16));
                write_bin(dump_dir + "/lin_bf16_bias.bin", bias_b.data(),
                          bias_b.size() * sizeof(__hip_bfloat16));
                write_bin(dump_dir + "/lin_bf16_out.bin", out_b.data(),
                          out_b.size() * sizeof(__hip_bfloat16));
            }
            hipFree(d_act_b);
            hipFree(d_bias_b);
            hipFree(d_out_b);
        }

        // --- M sweep ---------------------------------------------------------
        // The launcher switches M block at 16, 32 and 64 tokens, so every
        // boundary around those (and one past each) must be checked, not just
        // the round numbers.  Odd values exercise the partial-tile masking in
        // staging and the epilogue.
        {
            const int64_t mn = 128;
            for (int64_t m : {static_cast<int64_t>(15), static_cast<int64_t>(16),
                              static_cast<int64_t>(17), static_cast<int64_t>(31),
                              static_cast<int64_t>(32), static_cast<int64_t>(33),
                              static_cast<int64_t>(64), static_cast<int64_t>(65),
                              static_cast<int64_t>(100), static_cast<int64_t>(128),
                              static_cast<int64_t>(256)}) {
                std::vector<float> act(m * k);
                for (auto& v : act) {
                    v = normal(rng);
                }
                std::vector<__hip_bfloat16> act_b(m * k);
                std::vector<float> act_rounded(m * k);
                for (size_t i = 0; i < act.size(); ++i) {
                    act_b[i] = from_float<__hip_bfloat16>(act[i]);
                    act_rounded[i] = to_float<__hip_bfloat16>(act_b[i]);
                }
                std::vector<float> ref;
                ref_linear(act_rounded, got_codes, got_exponents, bits, m, mn, k,
                           nullptr, &ref);

                __hip_bfloat16* d_a = nullptr;
                __hip_bfloat16* d_o = nullptr;
                hipMalloc(&d_a, act_b.size() * sizeof(__hip_bfloat16));
                hipMalloc(&d_o, ref.size() * sizeof(__hip_bfloat16));
                hipMemcpy(d_a, act_b.data(), act_b.size() * sizeof(__hip_bfloat16),
                          hipMemcpyHostToDevice);
                const hipError_t st =
                    ttbf_linear(d_a, TTBF_DTYPE_BF16, d_codes, d_exp, bits, m, mn,
                                k, nullptr, d_o, nullptr);
                std::vector<__hip_bfloat16> outb(ref.size());
                hipMemcpy(outb.data(), d_o, outb.size() * sizeof(__hip_bfloat16),
                          hipMemcpyDeviceToHost);

                double abs_error = 0.0;
                double ref_mass = 0.0;
                for (size_t i = 0; i < outb.size(); ++i) {
                    const float rounded =
                        to_float<__hip_bfloat16>(from_float<__hip_bfloat16>(ref[i]));
                    abs_error += std::fabs(
                        static_cast<double>(to_float<__hip_bfloat16>(outb[i]) - rounded));
                    ref_mass += std::fabs(static_cast<double>(rounded));
                }
                const double nl1 = abs_error / std::max(1e-6, ref_mass);
                char label[96];
                std::snprintf(label, sizeof(label), "%s linear bf16 m=%lld norm_l1=%.3g",
                              bits == 4 ? "bf4" : "bf8", static_cast<long long>(m), nl1);
                report(st == hipSuccess && nl1 < 2e-3, label);
                hipFree(d_a);
                hipFree(d_o);
            }
        }

        // --- K sweep ---------------------------------------------------------
        // The API only promises ``k % 16 == 0``, but a thread stages a
        // ``kCodes``-wide span (16, 64 or 128 depending on the tile), so a k
        // that is not a multiple of the tile width ends the last tile
        // mid-span.  That used to zero the whole span, dropping the valid
        // leading columns with it.
        {
            const int64_t kn = 3;
            for (int64_t kk : {static_cast<int64_t>(16), static_cast<int64_t>(32),
                               static_cast<int64_t>(48), static_cast<int64_t>(80),
                               static_cast<int64_t>(144), static_cast<int64_t>(160),
                               static_cast<int64_t>(272)}) {
                std::vector<float> w(kn * kk);
                for (auto& v : w) {
                    v = normal(rng);
                }
                std::vector<uint8_t> kcodes;
                std::vector<int8_t> kexp;
                ref_quantize(w, bits, kn, kk, &kcodes, &kexp);

                uint8_t* dk_codes = nullptr;
                int8_t* dk_exp = nullptr;
                hipMalloc(&dk_codes, kcodes.size());
                hipMalloc(&dk_exp, kexp.size());
                hipMemcpy(dk_codes, kcodes.data(), kcodes.size(),
                          hipMemcpyHostToDevice);
                hipMemcpy(dk_exp, kexp.data(), kexp.size(),
                          hipMemcpyHostToDevice);

                bool k_ok = true;
                // One m per launcher tier, so every tile shape sees a partial K.
                for (int64_t m : {static_cast<int64_t>(2), static_cast<int64_t>(16),
                                  static_cast<int64_t>(17), static_cast<int64_t>(65)}) {
                    std::vector<float> act(m * kk);
                    for (auto& v : act) {
                        v = normal(rng);
                    }
                    std::vector<__hip_bfloat16> act_b(m * kk);
                    std::vector<float> act_rounded(m * kk);
                    for (size_t i = 0; i < act.size(); ++i) {
                        act_b[i] = from_float<__hip_bfloat16>(act[i]);
                        act_rounded[i] = to_float<__hip_bfloat16>(act_b[i]);
                    }
                    std::vector<float> ref;
                    ref_linear(act_rounded, kcodes, kexp, bits, m, kn, kk,
                               nullptr, &ref);

                    __hip_bfloat16* dk_a = nullptr;
                    __hip_bfloat16* dk_o = nullptr;
                    hipMalloc(&dk_a, act_b.size() * sizeof(__hip_bfloat16));
                    hipMalloc(&dk_o, ref.size() * sizeof(__hip_bfloat16));
                    hipMemcpy(dk_a, act_b.data(),
                              act_b.size() * sizeof(__hip_bfloat16),
                              hipMemcpyHostToDevice);
                    const hipError_t st = ttbf_linear(
                        dk_a, TTBF_DTYPE_BF16, dk_codes, dk_exp, bits, m, kn,
                        kk, nullptr, dk_o, nullptr);
                    std::vector<__hip_bfloat16> outb(ref.size());
                    hipMemcpy(outb.data(), dk_o,
                              outb.size() * sizeof(__hip_bfloat16),
                              hipMemcpyDeviceToHost);

                    double abs_error = 0.0;
                    double ref_mass = 0.0;
                    for (size_t i = 0; i < outb.size(); ++i) {
                        const float rounded = to_float<__hip_bfloat16>(
                            from_float<__hip_bfloat16>(ref[i]));
                        abs_error += std::fabs(static_cast<double>(
                            to_float<__hip_bfloat16>(outb[i]) - rounded));
                        ref_mass += std::fabs(static_cast<double>(rounded));
                    }
                    const double nl1 = abs_error / std::max(1e-6, ref_mass);
                    k_ok = k_ok && st == hipSuccess && nl1 < 2e-3;
                    hipFree(dk_a);
                    hipFree(dk_o);
                }
                char label[96];
                std::snprintf(label, sizeof(label), "%s linear partial-K k=%lld",
                              bits == 4 ? "bf4" : "bf8",
                              static_cast<long long>(kk));
                report(k_ok, label);
                hipFree(dk_codes);
                hipFree(dk_exp);
            }
        }

        // --- embedding gather ------------------------------------------------
        {
            const int64_t tokens = 16;
            std::vector<int64_t> ids(tokens);
            for (auto& id : ids) {
                id = static_cast<int64_t>(rng() % rows);
            }
            std::vector<float> ref(tokens * k);
            for (int64_t t = 0; t < tokens; ++t) {
                std::memcpy(ref.data() + t * k, grid.data() + ids[t] * k,
                            k * sizeof(float));
            }
            int64_t* d_ids = nullptr;
            float* d_emb = nullptr;
            hipMalloc(&d_ids, ids.size() * sizeof(int64_t));
            hipMalloc(&d_emb, ref.size() * sizeof(float));
            hipMemcpy(d_ids, ids.data(), ids.size() * sizeof(int64_t),
                      hipMemcpyHostToDevice);
            status = ttbf_embedding(d_ids, d_codes, d_exp, bits, tokens, rows, k,
                                    d_emb, TTBF_DTYPE_F32, nullptr);
            std::vector<float> out(ref.size());
            hipMemcpy(out.data(), d_emb, out.size() * sizeof(float),
                      hipMemcpyDeviceToHost);
            report(status == hipSuccess && out == ref,
                   bits == 4 ? "bf4 embedding gather bit-exact"
                             : "bf8 embedding gather bit-exact");
            if (!dump_dir.empty() && bits == 4) {
                write_bin(dump_dir + "/emb_ids.bin", ids.data(),
                          ids.size() * sizeof(int64_t));
                write_bin(dump_dir + "/emb_out.bin", out.data(),
                          out.size() * sizeof(float));
            }
            hipFree(d_ids);
            hipFree(d_emb);
        }

        hipFree(d_codes);
        hipFree(d_exp);
        hipFree(d_x);
        hipFree(d_grid);
        hipFree(d_grid_codes);
        hipFree(d_grid_exp);
        hipFree(d_failed);
    }

    // --- throughput ---------------------------------------------------------
    // Decode (m == 1) and prefill over a few token counts.  Prefill at small M
    // launches few CTAs (one per 128 output columns), so the M sweep shows how
    // much of the cost is occupancy rather than arithmetic.
    {
        const int64_t n = bench_n;
        const int64_t kk = bench_k;
        std::vector<uint8_t> codes(n * kk / 2);
        for (auto& b : codes) {
            b = static_cast<uint8_t>(rng() % 15);
        }
        std::vector<int8_t> exponents(n * kk / 16);
        for (auto& e : exponents) {
            e = static_cast<int8_t>(-8);
        }
        uint8_t* d_codes = nullptr;
        int8_t* d_exp = nullptr;
        hipMalloc(&d_codes, codes.size());
        hipMalloc(&d_exp, exponents.size());
        hipMemcpy(d_codes, codes.data(), codes.size(), hipMemcpyHostToDevice);
        hipMemcpy(d_exp, exponents.data(), exponents.size(),
                  hipMemcpyHostToDevice);

        auto time_it = [&](auto&& launch, int reps) {
            for (int rep = 0; rep < 5; ++rep) {
                launch();
            }
            hipDeviceSynchronize();
            double best = 1e9;
            for (int batch = 0; batch < 3; ++batch) {
                auto start = std::chrono::high_resolution_clock::now();
                for (int rep = 0; rep < reps; ++rep) {
                    launch();
                }
                hipDeviceSynchronize();
                const double seconds =
                    std::chrono::duration<double>(
                        std::chrono::high_resolution_clock::now() - start)
                        .count();
                best = std::min(best, seconds / reps);
            }
            return best;
        };

        // Decode.
        float* d_x = nullptr;
        float* d_out = nullptr;
        hipMalloc(&d_x, static_cast<size_t>(kk) * sizeof(float));
        hipMalloc(&d_out, static_cast<size_t>(n) * sizeof(float));
        const double decode_seconds = time_it(
            [&] {
                ttbf_linear(d_x, TTBF_DTYPE_F32, d_codes, d_exp, 4, 1, n, kk,
                            nullptr, d_out, nullptr);
            },
            100);
        std::printf(
            "decode %lldx%lld bf4: %.2f ms/call, %.1f GB/s packed-weight read\n",
            static_cast<long long>(n), static_cast<long long>(kk),
            decode_seconds * 1e3,
            static_cast<double>(codes.size()) / decode_seconds / 1e9);

        // Prefill over M.
        std::vector<int64_t> ms = {bench_m, 16, 32, 64, 128, 256, 512};
        std::sort(ms.begin(), ms.end());
        ms.erase(std::unique(ms.begin(), ms.end()), ms.end());
        for (int64_t pm : ms) {
            __hip_bfloat16* d_xb = nullptr;
            __hip_bfloat16* d_ob = nullptr;
            float* d_xf = nullptr;
            float* d_of = nullptr;
            hipMalloc(&d_xb, pm * kk * sizeof(__hip_bfloat16));
            hipMalloc(&d_ob, pm * n * sizeof(__hip_bfloat16));
            hipMalloc(&d_xf, pm * kk * sizeof(float));
            hipMalloc(&d_of, pm * n * sizeof(float));
            const double flops = 2.0 * static_cast<double>(pm) * n * kk;

            const double fp32_seconds = time_it(
                [&] {
                    ttbf_linear(d_xf, TTBF_DTYPE_F32, d_codes, d_exp, 4, pm, n,
                                kk, nullptr, d_of, nullptr);
                },
                20);
            const double bf16_seconds = time_it(
                [&] {
                    ttbf_linear(d_xb, TTBF_DTYPE_BF16, d_codes, d_exp, 4, pm, n,
                                kk, nullptr, d_ob, nullptr);
                },
                20);
            std::printf(
                "prefill m=%lld %lldx%lld bf4: fp32 %7.2f us %6.2f TFLOP/s | "
                "bf16 %7.2f us %6.2f TFLOP/s\n",
                static_cast<long long>(pm), static_cast<long long>(n),
                static_cast<long long>(kk), fp32_seconds * 1e6,
                flops / fp32_seconds / 1e12, bf16_seconds * 1e6,
                flops / bf16_seconds / 1e12);

            hipFree(d_xb);
            hipFree(d_ob);
            hipFree(d_xf);
            hipFree(d_of);
        }

        hipFree(d_codes);
        hipFree(d_exp);
        hipFree(d_x);
        hipFree(d_out);
    }

    std::printf("%s (%d failure%s)\n", failures == 0 ? "OK" : "FAILED", failures,
                failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
