// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0
// Standalone dispatch/algebra probe, not a performance or model-quality benchmark.
#include <oneapi/dnnl/dnnl.hpp>

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
template <class T>
void check(dnnl::memory::data_type dtype, T one, const char* label) {
    using namespace dnnl;
    constexpr int m = 256, n = 256, k = 512;
    engine cpu(engine::kind::cpu, 0);
    stream execution(cpu);
    std::vector<T> source(m * k, one), weights(k * n, one);
    std::vector<float> destination(m * n, 0);
    auto src = memory({{m, k}, dtype, memory::format_tag::ab}, cpu, source.data());
    auto wei = memory({{k, n}, dtype, memory::format_tag::ab}, cpu, weights.data());
    auto dst = memory({{m, n}, memory::data_type::f32, memory::format_tag::ab}, cpu, destination.data());
    auto descriptor = matmul::primitive_desc(cpu, src.get_desc(), wei.get_desc(), dst.get_desc());
    matmul operation(descriptor);
    operation.execute(execution, {{DNNL_ARG_SRC, src}, {DNNL_ARG_WEIGHTS, wei}, {DNNL_ARG_DST, dst}});
    execution.wait();
    for (float value : destination) {
        if (value != k) {
            throw std::runtime_error("matmul output differs from exact all-ones reference");
        }
    }
    std::cout << label << ": verified " << destination.size()
              << " outputs; implementation=" << descriptor.impl_info_str() << '\n';
}
}  // namespace

int main() {
    try {
        const auto* version = dnnl_version();
        std::cout << "oneDNN " << version->major << '.' << version->minor << '.' << version->patch << '\n';
        check<uint16_t>(dnnl::memory::data_type::bf16, 0x3f80, "BF16");
        check<int8_t>(dnnl::memory::data_type::s8, 1, "INT8");
        check<float>(dnnl::memory::data_type::f32, 1, "FP32");
    } catch (const std::exception& error) {
        std::cerr << "oneDNN probe failed: " << error.what() << '\n';
        return 1;
    }
}
