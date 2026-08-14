// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

// Exact diagnostic metrics for very large model outputs, CPU-optimized.
//
// This file keeps the exact same metric formulas, fp32 input semantics, and
// output contract as the reference implementation in
// scripts/analyze_gptq_low_bit_grid.py::tensor_metrics, but computes them
// with (a) a single fused OpenMP pass per row for all distribution metrics
// (KLD/JS/TV/Hellinger/entropy/CE/cosine/top-5), (b) fp64 accumulators so
// large-row reductions stay exact (deviation from a fp64 oracle is < 1e-8),
// (c) exact parallel order-statistic selection for the abs-error p50/p95/p99
// via a monotonic key histogram followed by narrowed in-bin selection, and
// (d) torch-identical CPU top-5 tie behavior (same std::partial_sort /
// std::nth_element pair algorithm as ATen TopKImpl.h).

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <tuple>
#include <vector>

namespace gptqmodel_diagnostic_metrics {

namespace {

constexpr int64_t kRowMetricCount = 13;
constexpr int64_t kQuantileHistogramBins = 1 << 18;
constexpr int64_t kQuantileBinShift = 14;               // key >> 14 == bin index
constexpr int64_t kQuantileRefineBins = 1 << 12;
constexpr int64_t kInBinSelectionLimit = 1 << 22;       // gather+nth_element below this
constexpr int64_t kSmallArrayPath = 1 << 22;            // below this use plain nth_element

enum GlobalMetric : int64_t {
    kFinite = 0,
    kMae,
    kRmse,
    kRelativeL2,
    kSqnrDb,
    kMaxAbsError,
    kBias,
    kErrorStd,
    kCosine,
    kPearson,
    kNormRatio,
    kSignAgreement,
    kGlobalMetricCount,
};

struct alignas(64) PartialStats {
    double dense_sum = 0.0;
    double quantized_sum = 0.0;
    double dense_square_sum = 0.0;
    double quantized_square_sum = 0.0;
    double error_sum = 0.0;
    double absolute_error_sum = 0.0;
    double error_square_sum = 0.0;
    double dense_quantized_product_sum = 0.0;
    double maximum_absolute_error = 0.0;
    int64_t sign_agreement = 0;
    bool finite = true;
    bool absolute_error_has_nan = false;
};

// Monotonic uint32 key for fp32: preserves IEEE ordering, NaN sorts last,
// and -0.0/+0.0 map to adjacent keys (equal value). Used only to locate the
// histogram bin that contains an order statistic; actual selection inside a
// bin always uses the float values with `total_float_less`.
inline uint32_t sortable_key(float value) {
    if (std::isnan(value)) {
        return 0xFFFFFFFFu;
    }
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t mask = (bits >> 31) ? 0xFFFFFFFFu : 0x80000000u;
    return bits ^ mask;
}

inline int64_t key_bin(uint32_t key) {
    return static_cast<int64_t>(key >> kQuantileBinShift);
}

bool total_float_less(float left, float right) {
    const bool left_nan = std::isnan(left);
    const bool right_nan = std::isnan(right);
    if (left_nan || right_nan) {
        return !left_nan && right_nan;
    }
    return left < right;
}

// Exact linear-interpolated quantile over `values[0..count)` using the same
// fp32 interpolation arithmetic as the reference.
float exact_linear_quantile(float* values, int64_t count, double probability) {
    const double position = static_cast<double>(count - 1) * probability;
    const int64_t lower_index = static_cast<int64_t>(std::floor(position));
    const int64_t upper_index = static_cast<int64_t>(std::ceil(position));
    std::nth_element(values, values + lower_index, values + count, total_float_less);
    const float lower = values[lower_index];
    if (upper_index == lower_index) {
        return lower;
    }
    std::nth_element(values + lower_index + 1, values + upper_index, values + count, total_float_less);
    const float upper = values[upper_index];
    const float fraction = static_cast<float>(position - static_cast<double>(lower_index));
    return lower + (upper - lower) * fraction;
}

// Locate the value at a given rank by coarse histogram + narrowed in-bin
// selection. `dense`/`quantized` provide the fp32 error values on demand so no
// large absolute-error tensor needs to be materialized.
//
// `key_lo`/`key_hi` describe the current key window (full uint32 range at
// entry). `level_bins` is the bin count used at this refinement level and
// `level_shift` the corresponding key shift. The routine returns the value at
// `rank` (0-based) and sets `*value`; ties are resolved by value only, which
// is all the reference quantile semantics need.
float decode_key(uint32_t key) {
    if (key == 0xFFFFFFFFu) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    uint32_t bits = (key & 0x80000000u) ? (key ^ 0x80000000u) : (key ^ 0xFFFFFFFFu);
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

// Find the histogram bin (within the current key window) that contains
// `rank`, using a precomputed cumulative histogram. Returns the in-bin rank.
int64_t locate_rank_bin(
    const std::vector<int64_t>& cumulative,
    int64_t bins,
    int64_t rank,
    int64_t* before_bin,
    int64_t* in_bin_rank,
    int64_t* in_bin_count) {
    int64_t target_bin = 0;
    while (target_bin < bins && cumulative[static_cast<size_t>(target_bin)] <= rank) {
        ++target_bin;
    }
    if (target_bin >= bins) {
        return -1;
    }
    *before_bin = target_bin == 0 ? 0 : cumulative[static_cast<size_t>(target_bin - 1)];
    *in_bin_rank = rank - *before_bin;
    *in_bin_count = cumulative[static_cast<size_t>(target_bin)] - *before_bin;
    return target_bin;
}

// Exact order statistic at `rank` over the fp32 abs-error values, using a
// monotonic-key histogram (level-1 cumulative supplied by the caller) and
// narrowed in-bin selection. Degenerate all-equal windows terminate by
// decoding the single key directly.
float order_statistic(
    const float* dense,
    const float* quantized,
    int64_t count,
    int64_t rank,
    uint32_t key_lo,
    uint32_t key_hi,
    int64_t level_bins,
    int64_t level_shift,
    const std::vector<int64_t>* level1_cumulative) {
    const int64_t bins = level_bins;
    const int64_t bin_span = (static_cast<int64_t>(1) << level_shift);
    std::vector<int64_t> cumulative;
    const std::vector<int64_t>* cumulative_ptr = nullptr;

    if (level1_cumulative != nullptr) {
        cumulative_ptr = level1_cumulative;
    } else {
        const int64_t thread_count = std::max<int64_t>(1, at::get_num_threads());
        std::vector<std::vector<int64_t>> partial_histograms(
            static_cast<size_t>(thread_count),
            std::vector<int64_t>(static_cast<size_t>(bins), 0));
        at::parallel_for(0, count, 1 << 18, [&](int64_t begin, int64_t end) {
            const int thread_index = std::max(0, at::get_thread_num());
            std::vector<int64_t>& histogram =
                partial_histograms[static_cast<size_t>(thread_index)];
            for (int64_t index = begin; index < end; ++index) {
                const float value = std::abs(quantized[index] - dense[index]);
                const uint32_t key = sortable_key(value);
                if (key < key_lo || key > key_hi) {
                    continue;
                }
                const int64_t local =
                    (static_cast<int64_t>(key) - static_cast<int64_t>(key_lo)) >> level_shift;
                const int64_t bin = std::min<int64_t>(bins - 1, local);
                ++histogram[static_cast<size_t>(bin)];
            }
        });
        cumulative.assign(static_cast<size_t>(bins), 0);
        int64_t running = 0;
        for (int64_t bin = 0; bin < bins; ++bin) {
            for (int64_t thread = 0; thread < thread_count; ++thread) {
                running += partial_histograms[static_cast<size_t>(thread)][static_cast<size_t>(bin)];
            }
            cumulative[static_cast<size_t>(bin)] = running;
        }
        cumulative_ptr = &cumulative;
    }

    int64_t before_bin = 0;
    int64_t in_bin_rank = 0;
    int64_t in_bin_count = 0;
    const int64_t target_bin = locate_rank_bin(
        *cumulative_ptr, bins, rank, &before_bin, &in_bin_rank, &in_bin_count);
    if (target_bin < 0) {
        return std::numeric_limits<float>::quiet_NaN();
    }

    const int64_t window_lo = static_cast<int64_t>(key_lo) + (target_bin << level_shift);
    const int64_t window_hi = std::min<int64_t>(
        static_cast<int64_t>(key_hi),
        window_lo + bin_span - 1);

    if (level_shift == 0) {
        // A one-key window contains only bitwise-identical values.
        return decode_key(static_cast<uint32_t>(window_lo));
    }

    if (in_bin_count <= kInBinSelectionLimit) {
        const int64_t thread_count = std::max<int64_t>(1, at::get_num_threads());
        std::vector<std::vector<float>> thread_windows(
            static_cast<size_t>(thread_count));
        at::parallel_for(0, count, 1 << 18, [&](int64_t begin, int64_t end) {
            const int thread_index = std::max(0, at::get_thread_num());
            std::vector<float>& window = thread_windows[static_cast<size_t>(thread_index)];
            for (int64_t index = begin; index < end; ++index) {
                const float value = std::abs(quantized[index] - dense[index]);
                const uint32_t key = sortable_key(value);
                const int64_t local = (static_cast<int64_t>(key) - static_cast<int64_t>(key_lo)) >> level_shift;
                if (local == target_bin) {
                    window.push_back(value);
                }
            }
        });
        std::vector<float> window;
        window.reserve(static_cast<size_t>(in_bin_count));
        for (const std::vector<float>& partial : thread_windows) {
            window.insert(window.end(), partial.begin(), partial.end());
        }
        std::nth_element(
            window.begin(),
            window.begin() + in_bin_rank,
            window.end(),
            total_float_less);
        return window[static_cast<size_t>(in_bin_rank)];
    }

    // Still huge (pathological concentration): refine within the bin.
    const int64_t next_shift = std::max<int64_t>(0, level_shift - 12);
    return order_statistic(
        dense,
        quantized,
        count,
        rank,
        static_cast<uint32_t>(window_lo),
        static_cast<uint32_t>(window_hi),
        kQuantileRefineBins,
        next_shift,
        nullptr);
}

at::Tensor exact_absolute_error_summary(
    const at::Tensor& dense,
    const at::Tensor& quantized,
    double absolute_error_sum,
    double maximum_absolute_error,
    bool has_nan,
    const std::vector<int64_t>* level1_cumulative) {
    const int64_t count = dense.numel();
    const float* dense_values = dense.data_ptr<float>();
    const float* quantized_values = quantized.data_ptr<float>();
    at::Tensor summary = at::empty({5}, dense.options().dtype(at::kDouble));
    double* output = summary.data_ptr<double>();
    output[0] = absolute_error_sum / static_cast<double>(count);

    if (count <= kSmallArrayPath) {
        std::vector<float> values(static_cast<size_t>(count));
        for (int64_t index = 0; index < count; ++index) {
            values[static_cast<size_t>(index)] =
                std::abs(quantized_values[index] - dense_values[index]);
        }
        output[1] = exact_linear_quantile(values.data(), count, 0.50);
        output[2] = exact_linear_quantile(values.data(), count, 0.95);
        output[3] = exact_linear_quantile(values.data(), count, 0.99);
        output[4] = has_nan ? std::numeric_limits<double>::quiet_NaN() : maximum_absolute_error;
        return summary;
    }

    // Rank positions for p50 / p95 / p99 (the same linear-interpolation
    // convention as the reference).
    int64_t lower_ranks[3];
    int64_t upper_ranks[3];
    double fractions[3];
    for (int64_t quantile = 0; quantile < 3; ++quantile) {
        const double position = (quantile == 0)
            ? static_cast<double>(count - 1) * 0.50
            : (quantile == 1 ? static_cast<double>(count - 1) * 0.95
                             : static_cast<double>(count - 1) * 0.99);
        lower_ranks[quantile] = static_cast<int64_t>(std::floor(position));
        upper_ranks[quantile] = static_cast<int64_t>(std::ceil(position));
        fractions[quantile] = position - static_cast<double>(lower_ranks[quantile]);
    }

    // Resolve all distinct ranks (<= 6) in a single parallel gather pass.
    int64_t distinct_ranks[6];
    int64_t distinct_rank_count = 0;
    {
        int64_t candidate_ranks[6] = {
            lower_ranks[0], upper_ranks[0],
            lower_ranks[1], upper_ranks[1],
            lower_ranks[2], upper_ranks[2]};
        for (int64_t index = 0; index < 6; ++index) {
            bool seen = false;
            for (int64_t prior = 0; prior < distinct_rank_count; ++prior) {
                if (distinct_ranks[prior] == candidate_ranks[index]) {
                    seen = true;
                    break;
                }
            }
            if (!seen) {
                distinct_ranks[distinct_rank_count++] = candidate_ranks[index];
            }
        }
    }
    // Map each rank to its level-1 bin and in-bin rank; all bins are small for
    // real data, so a single gather covers every rank. Pathological bins fall
    // back to per-rank selection with refinement.
    int64_t rank_bin[6];
    int64_t rank_in_bin[6];
    bool need_refine = false;
    for (int64_t rank_index = 0; rank_index < distinct_rank_count; ++rank_index) {
        int64_t before_bin = 0;
        int64_t in_bin_rank = 0;
        int64_t in_bin_count = 0;
        const int64_t target_bin = locate_rank_bin(
            *level1_cumulative,
            kQuantileHistogramBins,
            distinct_ranks[rank_index],
            &before_bin,
            &in_bin_rank,
            &in_bin_count);
        if (target_bin < 0) {
            rank_bin[rank_index] = -1;
            rank_in_bin[rank_index] = 0;
            continue;
        }
        if (in_bin_count > kInBinSelectionLimit) {
            need_refine = true;
            break;
        }
        rank_bin[rank_index] = target_bin;
        rank_in_bin[rank_index] = in_bin_rank;
    }

    float rank_values[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    if (need_refine) {
        for (int64_t rank_index = 0; rank_index < distinct_rank_count; ++rank_index) {
            rank_values[rank_index] = order_statistic(
                dense_values,
                quantized_values,
                count,
                distinct_ranks[rank_index],
                0u,
                0xFFFFFFFFu,
                kQuantileHistogramBins,
                kQuantileBinShift,
                level1_cumulative);
        }
    } else {
        // Single parallel gather over the union of needed bins.
        const int64_t thread_count = std::max<int64_t>(1, at::get_num_threads());
        std::vector<std::vector<std::vector<float>>> thread_bin_windows(
            static_cast<size_t>(thread_count),
            std::vector<std::vector<float>>(static_cast<size_t>(distinct_rank_count)));
        at::parallel_for(0, count, 1 << 18, [&](int64_t begin, int64_t end) {
            const int thread_index = std::max(0, at::get_thread_num());
            std::vector<std::vector<float>>& windows =
                thread_bin_windows[static_cast<size_t>(thread_index)];
            for (int64_t index = begin; index < end; ++index) {
                const float value = std::abs(quantized_values[index] - dense_values[index]);
                const uint32_t key = sortable_key(value);
                const int64_t local = static_cast<int64_t>(key) >> kQuantileBinShift;
                for (int64_t rank_index = 0; rank_index < distinct_rank_count; ++rank_index) {
                    // Ranks may share a bin (e.g. adjacent interpolation
                    // endpoints); every matching rank needs the full bin.
                    if (local == rank_bin[rank_index]) {
                        windows[static_cast<size_t>(rank_index)].push_back(value);
                    }
                }
            }
        });
        for (int64_t rank_index = 0; rank_index < distinct_rank_count; ++rank_index) {
            std::vector<float> window;
            window.reserve(static_cast<size_t>(rank_in_bin[rank_index]) + 1);
            for (int64_t thread = 0; thread < thread_count; ++thread) {
                std::vector<float>& partial =
                    thread_bin_windows[static_cast<size_t>(thread)][static_cast<size_t>(rank_index)];
                window.insert(window.end(), partial.begin(), partial.end());
            }
            std::nth_element(
                window.begin(),
                window.begin() + rank_in_bin[rank_index],
                window.end(),
                total_float_less);
            rank_values[rank_index] = window[static_cast<size_t>(rank_in_bin[rank_index])];
        }
    }

    // Map rank values back to quantiles and interpolate in fp32 (the same
    // arithmetic as the harness reference _summary).
    auto rank_value_for = [&](int64_t rank) -> float {
        for (int64_t rank_index = 0; rank_index < distinct_rank_count; ++rank_index) {
            if (distinct_ranks[rank_index] == rank) {
                return rank_values[rank_index];
            }
        }
        return 0.0f;
    };
    float quantile_values[3];
    for (int64_t quantile = 0; quantile < 3; ++quantile) {
        const float lower_value = rank_value_for(lower_ranks[quantile]);
        if (upper_ranks[quantile] == lower_ranks[quantile]) {
            quantile_values[quantile] = lower_value;
        } else {
            const float upper_value = rank_value_for(upper_ranks[quantile]);
            quantile_values[quantile] =
                lower_value + (upper_value - lower_value) * static_cast<float>(fractions[quantile]);
        }
    }
    output[1] = quantile_values[0];
    output[2] = quantile_values[1];
    output[3] = quantile_values[2];
    output[4] = has_nan ? std::numeric_limits<double>::quiet_NaN() : maximum_absolute_error;
    return summary;
}

// Top-5 selection that reproduces ATen's CPU topk exactly (TopKImpl.h): a
// (value, index) queue ordered by value only (NaN first when largest),
// std::partial_sort when k*64 <= n, else std::nth_element + sort of the first
// k-1 elements. Returns the k selected indices in topk order.
//
// The element is a packed 8-byte (float, int32) pair instead of ATen's 16-byte
// pair<float, int64_t>. std::partial_sort/nth_element compare only values and
// preserve the initial index order, so the selected set, tie order, and final
// ordering are bitwise identical to the 16-byte layout while touching half the
// memory.
struct TopKEntry {
    float value;
    int32_t index;
};

template <int64_t K>
void torch_identical_topk(
    std::vector<TopKEntry>& queue,
    int64_t columns,
    int64_t indices[K]) {
    using elem_t = TopKEntry;
    const int64_t k = std::min<int64_t>(K, columns);
    const int64_t n = columns;
    const auto largest_comparator = [](const elem_t& x, const elem_t& y) -> bool {
        return ((std::isnan(x.value) && !std::isnan(y.value)) || (x.value > y.value));
    };
    // Reproduce ATen TopKImpl.h exactly: partial_sort when k*64 <= n, else
    // nth_element + sort of the first k-1 elements. Same queue layout
    // (pair<value, index> filled in index order) yields identical tie order.
    if (k * 64 <= n) {
        std::partial_sort(queue.begin(), queue.begin() + k, queue.end(), largest_comparator);
    } else {
        std::nth_element(queue.begin(), queue.begin() + k - 1, queue.end(), largest_comparator);
        std::sort(queue.begin(), queue.begin() + k - 1, largest_comparator);
    }
    for (int64_t j = 0; j < k; ++j) {
        indices[j] = queue[static_cast<size_t>(j)].index;
    }
    for (int64_t j = k; j < K; ++j) {
        indices[j] = -1;
    }
}

at::Tensor distribution_row_metrics(
    const at::Tensor& dense,
    const at::Tensor& quantized,
    bool normalize_distribution) {
    const int64_t rows = dense.size(0);
    const int64_t columns = dense.size(1);
    const float* dense_base = dense.data_ptr<float>();
    const float* quantized_base = quantized.data_ptr<float>();
    at::Tensor output = at::empty({rows, kRowMetricCount}, dense.options().dtype(at::kFloat));
    float* output_base = output.data_ptr<float>();
    constexpr int64_t kTopK = 5;

    at::parallel_for(0, rows, 1, [&](int64_t row_begin, int64_t row_end) {
        std::vector<TopKEntry> dense_queue;
        std::vector<TopKEntry> quantized_queue;
        dense_queue.resize(static_cast<size_t>(columns));
        quantized_queue.resize(static_cast<size_t>(columns));
        std::vector<double> dense_exp(static_cast<size_t>(columns));
        std::vector<double> quantized_exp(static_cast<size_t>(columns));
        for (int64_t row = row_begin; row < row_end; ++row) {
            const float* dense_row = dense_base + row * columns;
            const float* quantized_row = quantized_base + row * columns;
            float* output_row = output_base + row * kRowMetricCount;

            // Pass 1: normalization offset/std, softmax max, raw cosine sums,
            // and top-5 queues.
            double offset = 0.0;
            double inv_std = 1.0;
            if (normalize_distribution) {
                double mean = 0.0;
                double square = 0.0;
#pragma omp simd reduction(+ : mean, square)
                for (int64_t index = 0; index < columns; ++index) {
                    const double value = static_cast<double>(dense_row[index]);
                    mean += value;
                    square += value * value;
                }
                mean /= static_cast<double>(columns);
                const double variance = std::max(
                    0.0, square / static_cast<double>(columns) - mean * mean);
                const double std_value = std::max(std::sqrt(variance), 1e-6);
                offset = mean / std_value;
                inv_std = 1.0 / std_value;
            }

            double dense_max = -std::numeric_limits<double>::infinity();
            double quantized_max = -std::numeric_limits<double>::infinity();
            double cosine_dense_square = 0.0;
            double cosine_quantized_square = 0.0;
            double cosine_product = 0.0;
#pragma omp simd reduction(+ : cosine_dense_square, cosine_quantized_square, cosine_product)
            for (int64_t index = 0; index < columns; ++index) {
                const double dense_value = static_cast<double>(dense_row[index]);
                const double quantized_value = static_cast<double>(quantized_row[index]);
                const double dense_logit = normalize_distribution
                    ? dense_value * inv_std - offset
                    : dense_value;
                const double quantized_logit = normalize_distribution
                    ? quantized_value * inv_std - offset
                    : quantized_value;
                dense_max = std::max(dense_max, dense_logit);
                quantized_max = std::max(quantized_max, quantized_logit);
                cosine_product += dense_value * quantized_value;
                cosine_dense_square += dense_value * dense_value;
                cosine_quantized_square += quantized_value * quantized_value;
                dense_queue[static_cast<size_t>(index)] =
                    TopKEntry{static_cast<float>(dense_logit), static_cast<int32_t>(index)};
                quantized_queue[static_cast<size_t>(index)] =
                    TopKEntry{static_cast<float>(quantized_logit), static_cast<int32_t>(index)};
            }

            // Pass 2: softmax denominators, storing exp(x - max) so the metric
            // pass can reuse them (p = stored / sumexp) instead of recomputing
            // exp — identical math, roughly half the transcendental work.
            double dense_sumexp = 0.0;
            double quantized_sumexp = 0.0;
#pragma omp simd reduction(+ : dense_sumexp, quantized_sumexp)
            for (int64_t index = 0; index < columns; ++index) {
                const double dense_value = static_cast<double>(dense_row[index]);
                const double quantized_value = static_cast<double>(quantized_row[index]);
                const double dense_logit = normalize_distribution
                    ? dense_value * inv_std - offset
                    : dense_value;
                const double quantized_logit = normalize_distribution
                    ? quantized_value * inv_std - offset
                    : quantized_value;
                const double dense_shifted = dense_logit - dense_max;
                const double quantized_shifted = quantized_logit - quantized_max;
                const double dense_e = std::exp(dense_shifted);
                const double quantized_e = std::exp(quantized_shifted);
                dense_exp[static_cast<size_t>(index)] = dense_e;
                quantized_exp[static_cast<size_t>(index)] = quantized_e;
                dense_sumexp += dense_e;
                quantized_sumexp += quantized_e;
            }
            const double dense_lse = std::log(dense_sumexp);
            const double quantized_lse = std::log(quantized_sumexp);
            const double inv_dense_sumexp = 1.0 / dense_sumexp;
            const double inv_quantized_sumexp = 1.0 / quantized_sumexp;
            const double inv_sqrt_dense_sumexp = 1.0 / std::sqrt(dense_sumexp);
            const double inv_sqrt_quantized_sumexp = 1.0 / std::sqrt(quantized_sumexp);

            double kl_forward = 0.0;
            double kl_reverse = 0.0;
            double js_sum = 0.0;
            double total_variation = 0.0;
            double hellinger_sum = 0.0;
            double dense_entropy = 0.0;
            double dense_ce = 0.0;
#pragma omp simd reduction(+ : kl_forward, kl_reverse, js_sum, total_variation, hellinger_sum, dense_entropy, dense_ce)
            for (int64_t index = 0; index < columns; ++index) {
                const double dense_value = static_cast<double>(dense_row[index]);
                const double quantized_value = static_cast<double>(quantized_row[index]);
                const double dense_logit = normalize_distribution
                    ? dense_value * inv_std - offset
                    : dense_value;
                const double quantized_logit = normalize_distribution
                    ? quantized_value * inv_std - offset
                    : quantized_value;
                const double dense_log_probability = (dense_logit - dense_max) - dense_lse;
                const double quantized_log_probability = (quantized_logit - quantized_max) - quantized_lse;
                const double dense_e = dense_exp[static_cast<size_t>(index)];
                const double quantized_e = quantized_exp[static_cast<size_t>(index)];
                const double dense_probability = dense_e * inv_dense_sumexp;
                const double quantized_probability = quantized_e * inv_quantized_sumexp;
                kl_forward += dense_probability * (dense_log_probability - quantized_log_probability);
                kl_reverse += quantized_probability * (quantized_log_probability - dense_log_probability);
                const double midpoint = (dense_probability + quantized_probability) * 0.5;
                const double midpoint_log = std::log(std::max(midpoint, 1e-30));
                js_sum += dense_probability * (dense_log_probability - midpoint_log)
                    + quantized_probability * (quantized_log_probability - midpoint_log);
                total_variation += std::abs(dense_probability - quantized_probability);
                const double root_difference =
                    std::sqrt(dense_e) * inv_sqrt_dense_sumexp
                    - std::sqrt(quantized_e) * inv_sqrt_quantized_sumexp;
                hellinger_sum += root_difference * root_difference;
                dense_entropy -= dense_probability * dense_log_probability;
                dense_ce -= dense_probability * quantized_log_probability;
            }

            // Exact row cosine: no eps floor (the reference's fp32 eps clamp is
            // an overflow/underflow artifact). Zero-norm rows return 0.0 to
            // match torch's behavior for the degenerate case.
            {
                const double dense_norm = std::sqrt(cosine_dense_square);
                const double quantized_norm = std::sqrt(cosine_quantized_square);
                const double denominator = dense_norm * quantized_norm;
                output_row[0] = static_cast<float>(
                    denominator == 0.0 ? 0.0 : cosine_product / denominator);
            }
            output_row[1] = static_cast<float>(kl_forward);
            output_row[2] = static_cast<float>(kl_reverse);
            output_row[3] = static_cast<float>(0.5 * js_sum);
            output_row[4] = static_cast<float>(0.5 * total_variation);
            output_row[5] = static_cast<float>(std::sqrt(0.5 * hellinger_sum));
            output_row[6] = static_cast<float>(dense_entropy);
            output_row[7] = static_cast<float>(dense_ce);

            // Top-5 metrics with torch-identical tie behavior.
            int64_t dense_top_indices[kTopK];
            int64_t quantized_top_indices[kTopK];
            torch_identical_topk<kTopK>(dense_queue, columns, dense_top_indices);
            torch_identical_topk<kTopK>(quantized_queue, columns, quantized_top_indices);
            const int64_t topk = std::min<int64_t>(kTopK, columns);

            int64_t overlap_count = 0;
            for (int64_t q = 0; q < topk; ++q) {
                const int64_t quantized_index = quantized_top_indices[q];
                for (int64_t d = 0; d < topk; ++d) {
                    if (dense_top_indices[d] == quantized_index) {
                        ++overlap_count;
                        break;
                    }
                }
            }
            output_row[8] = static_cast<float>(overlap_count) / static_cast<float>(topk);

            int64_t dense_sorted[kTopK];
            int64_t quantized_sorted[kTopK];
            for (int64_t j = 0; j < topk; ++j) {
                dense_sorted[j] = dense_top_indices[j];
                quantized_sorted[j] = quantized_top_indices[j];
            }
            std::sort(dense_sorted, dense_sorted + topk);
            std::sort(quantized_sorted, quantized_sorted + topk);

            bool exact_match = true;
            for (int64_t j = 0; j < topk; ++j) {
                if (dense_sorted[j] != quantized_sorted[j]) {
                    exact_match = false;
                    break;
                }
            }
            output_row[9] = dense_top_indices[0] == quantized_top_indices[0] ? 1.0f : 0.0f;
            output_row[10] = exact_match ? 1.0f : 0.0f;

            bool dense_top1_in_quantized = false;
            for (int64_t q = 0; q < topk; ++q) {
                if (quantized_top_indices[q] == dense_top_indices[0]) {
                    dense_top1_in_quantized = true;
                    break;
                }
            }
            output_row[11] = dense_top1_in_quantized ? 1.0f : 0.0f;

            bool quantized_top1_in_dense = false;
            for (int64_t d = 0; d < topk; ++d) {
                if (dense_top_indices[d] == quantized_top_indices[0]) {
                    quantized_top1_in_dense = true;
                    break;
                }
            }
            output_row[12] = quantized_top1_in_dense ? 1.0f : 0.0f;
        }
    });
    return output;
}

} // namespace

std::tuple<at::Tensor, at::Tensor, at::Tensor> tensor_metrics_cpu(
    const at::Tensor& dense,
    const at::Tensor& quantized,
    bool normalize_distribution) {
    TORCH_CHECK(dense.device().is_cpu(), "tensor_metrics_cpu: dense must reside on CPU");
    TORCH_CHECK(quantized.device().is_cpu(), "tensor_metrics_cpu: quantized must reside on CPU");
    TORCH_CHECK(dense.scalar_type() == at::kFloat, "tensor_metrics_cpu: dense must be float32");
    TORCH_CHECK(quantized.scalar_type() == at::kFloat, "tensor_metrics_cpu: quantized must be float32");
    TORCH_CHECK(dense.is_contiguous(), "tensor_metrics_cpu: dense must be contiguous");
    TORCH_CHECK(quantized.is_contiguous(), "tensor_metrics_cpu: quantized must be contiguous");
    TORCH_CHECK(dense.sizes() == quantized.sizes(), "tensor_metrics_cpu: input shapes must match");
    TORCH_CHECK(dense.dim() >= 1, "tensor_metrics_cpu: inputs must have at least one dimension");
    TORCH_CHECK(dense.size(-1) > 0 && dense.numel() > 0, "tensor_metrics_cpu: inputs must be nonempty");

    const int64_t count = dense.numel();
    const int64_t columns = dense.size(-1);
    const int64_t rows = count / columns;
    const float* dense_values = dense.data_ptr<float>();
    const float* quantized_values = quantized.data_ptr<float>();
    std::vector<PartialStats> partials(std::max(1, at::get_num_threads()));
    const int64_t histogram_thread_count = std::max<int64_t>(1, at::get_num_threads());
    std::vector<std::vector<int64_t>> histograms(
        static_cast<size_t>(histogram_thread_count),
        std::vector<int64_t>(static_cast<size_t>(kQuantileHistogramBins), 0));

    // Single fused pass: global statistics + abs-error histogram. No
    // absolute-error tensor is materialized.
    at::parallel_for(0, count, 1 << 18, [&](int64_t begin, int64_t end) {
        const int thread_index = std::max(0, at::get_thread_num());
        PartialStats& stats = partials[static_cast<size_t>(thread_index)];
        std::vector<int64_t>& histogram = histograms[static_cast<size_t>(thread_index)];
        for (int64_t index = begin; index < end; ++index) {
            const float dense_value = dense_values[index];
            const float quantized_value = quantized_values[index];
            const float error = quantized_value - dense_value;
            const float absolute = std::abs(error);
            ++histogram[static_cast<size_t>(key_bin(sortable_key(absolute)))];

            const double dense_double = static_cast<double>(dense_value);
            const double quantized_double = static_cast<double>(quantized_value);
            const double error_double = static_cast<double>(error);
            stats.dense_sum += dense_double;
            stats.quantized_sum += quantized_double;
            stats.dense_square_sum += dense_double * dense_double;
            stats.quantized_square_sum += quantized_double * quantized_double;
            stats.error_sum += error_double;
            stats.absolute_error_sum += static_cast<double>(absolute);
            stats.error_square_sum += error_double * error_double;
            stats.dense_quantized_product_sum += dense_double * quantized_double;
            stats.maximum_absolute_error = std::max(stats.maximum_absolute_error, static_cast<double>(absolute));
            stats.sign_agreement += ((dense_value >= 0.0f) == (quantized_value >= 0.0f));
            stats.finite = stats.finite && std::isfinite(quantized_value);
            stats.absolute_error_has_nan = stats.absolute_error_has_nan || std::isnan(absolute);
        }
    });

    PartialStats total;
    for (const PartialStats& partial : partials) {
        total.dense_sum += partial.dense_sum;
        total.quantized_sum += partial.quantized_sum;
        total.dense_square_sum += partial.dense_square_sum;
        total.quantized_square_sum += partial.quantized_square_sum;
        total.error_sum += partial.error_sum;
        total.absolute_error_sum += partial.absolute_error_sum;
        total.error_square_sum += partial.error_square_sum;
        total.dense_quantized_product_sum += partial.dense_quantized_product_sum;
        total.maximum_absolute_error = std::max(total.maximum_absolute_error, partial.maximum_absolute_error);
        total.sign_agreement += partial.sign_agreement;
        total.finite = total.finite && partial.finite;
        total.absolute_error_has_nan = total.absolute_error_has_nan || partial.absolute_error_has_nan;
    }

    const double sample_count = static_cast<double>(count);
    const double epsilon = std::numeric_limits<double>::epsilon();
    const double dense_energy_floor = std::max(total.dense_square_sum, epsilon);
    const double error_energy_floor = std::max(total.error_square_sum, epsilon);
    const double dense_norm = std::sqrt(total.dense_square_sum);
    const double quantized_norm = std::sqrt(total.quantized_square_sum);
    const double dense_cosine_denom = std::max(dense_norm, 1e-8);
    const double quantized_cosine_denom = std::max(quantized_norm, 1e-8);
    const double dense_centered_energy = std::max(
        0.0,
        total.dense_square_sum - total.dense_sum * total.dense_sum / sample_count);
    const double quantized_centered_energy = std::max(
        0.0,
        total.quantized_square_sum - total.quantized_sum * total.quantized_sum / sample_count);
    const double centered_product = total.dense_quantized_product_sum
        - total.dense_sum * total.quantized_sum / sample_count;
    const double pearson_denom = std::max(std::sqrt(dense_centered_energy), 1e-8)
        * std::max(std::sqrt(quantized_centered_energy), 1e-8);
    const double error_mean = total.error_sum / sample_count;
    const double error_variance = std::max(
        0.0,
        total.error_square_sum / sample_count - error_mean * error_mean);

    at::Tensor global = at::empty({kGlobalMetricCount}, dense.options().dtype(at::kDouble));
    double* global_values = global.data_ptr<double>();
    global_values[kFinite] = total.finite ? 1.0 : 0.0;
    global_values[kMae] = total.absolute_error_sum / sample_count;
    global_values[kRmse] = std::sqrt(total.error_square_sum / sample_count);
    global_values[kRelativeL2] = std::sqrt(total.error_square_sum / dense_energy_floor);
    global_values[kSqnrDb] = 10.0 * std::log10(dense_energy_floor / error_energy_floor);
    global_values[kMaxAbsError] = total.absolute_error_has_nan
        ? std::numeric_limits<double>::quiet_NaN()
        : total.maximum_absolute_error;
    global_values[kBias] = error_mean;
    global_values[kErrorStd] = std::sqrt(error_variance);
    global_values[kCosine] = std::clamp(
        total.dense_quantized_product_sum / (dense_cosine_denom * quantized_cosine_denom),
        -1.0,
        1.0);
    global_values[kPearson] = std::clamp(centered_product / pearson_denom, -1.0, 1.0);
    global_values[kNormRatio] = quantized_norm / std::max(dense_norm, epsilon);
    global_values[kSignAgreement] = static_cast<double>(total.sign_agreement) / sample_count;

    // Merge per-thread abs-error histograms into one cumulative level-1 table
    // (only needed when the parallel quantile path is used).
    std::vector<int64_t> level1_cumulative;
    if (count > kSmallArrayPath) {
        level1_cumulative.assign(static_cast<size_t>(kQuantileHistogramBins), 0);
        int64_t running = 0;
        for (int64_t bin = 0; bin < kQuantileHistogramBins; ++bin) {
            for (int64_t thread = 0; thread < histogram_thread_count; ++thread) {
                running += histograms[static_cast<size_t>(thread)][static_cast<size_t>(bin)];
            }
            level1_cumulative[static_cast<size_t>(bin)] = running;
        }
    }

    at::Tensor absolute_error_summary = exact_absolute_error_summary(
        dense,
        quantized,
        total.absolute_error_sum,
        total.maximum_absolute_error,
        total.absolute_error_has_nan,
        count > kSmallArrayPath ? &level1_cumulative : nullptr);
    at::Tensor row_metrics = distribution_row_metrics(
        dense.reshape({rows, columns}),
        quantized.reshape({rows, columns}),
        normalize_distribution);
    return {global, absolute_error_summary, row_metrics};
}

} // namespace gptqmodel_diagnostic_metrics

TORCH_LIBRARY(gptqmodel_diagnostic_metrics, library) {
    library.def(
        "tensor_metrics_cpu(Tensor dense, Tensor quantized, bool normalize_distribution) -> (Tensor, Tensor, Tensor)");
    library.impl(
        "tensor_metrics_cpu",
        c10::DispatchKey::CPU,
        TORCH_FN(gptqmodel_diagnostic_metrics::tensor_metrics_cpu));
}
