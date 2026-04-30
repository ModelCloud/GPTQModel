// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include <torch/extension.h>

#include <c10/core/DeviceType.h>
#include <c10/util/Optional.h>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstdint>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "acl/acl_base.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_weight_quant_batch_matmul_v3.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace {

constexpr const char* kExecutorCacheEnv = "GPTQMODEL_KOMODO_CANN_V3_EXECUTOR_CACHE";
constexpr const char* kWorkspaceCacheEnv = "GPTQMODEL_KOMODO_CANN_V3_WORKSPACE_CACHE";
constexpr const char* kInnerPreciseEnv = "GPTQMODEL_KOMODO_CANN_INNER_PRECISE";

class AclTensorHandle {
public:
    AclTensorHandle() = default;
    explicit AclTensorHandle(aclTensor* tensor) : tensor_(tensor) {}
    AclTensorHandle(const AclTensorHandle&) = delete;
    AclTensorHandle& operator=(const AclTensorHandle&) = delete;
    AclTensorHandle(AclTensorHandle&& other) noexcept : tensor_(other.tensor_) { other.tensor_ = nullptr; }
    AclTensorHandle& operator=(AclTensorHandle&& other) noexcept
    {
        if (this != &other) {
            reset();
            tensor_ = other.tensor_;
            other.tensor_ = nullptr;
        }
        return *this;
    }
    ~AclTensorHandle() { reset(); }

    aclTensor* get() const { return tensor_; }

private:
    void reset()
    {
        if (tensor_ != nullptr) {
            aclDestroyTensor(tensor_);
            tensor_ = nullptr;
        }
    }

    aclTensor* tensor_ = nullptr;
};

std::vector<int64_t> contiguous_strides(const std::vector<int64_t>& sizes)
{
    std::vector<int64_t> strides(sizes.size(), 1);
    for (int64_t i = static_cast<int64_t>(sizes.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * sizes[i + 1];
    }
    return strides;
}

aclDataType to_acl_dtype(at::ScalarType dtype)
{
    switch (dtype) {
        case at::kHalf:
            return ACL_FLOAT16;
        case at::kFloat:
            return ACL_FLOAT;
        case at::kInt:
            return ACL_INT32;
        case at::kByte:
            return ACL_UINT8;
        default:
            TORCH_CHECK(false, "Unsupported dtype for Komodo-CANN V3 probe: ", dtype);
    }
}

AclTensorHandle make_acl_tensor(
    const at::Tensor& tensor,
    aclDataType dtype,
    const std::vector<int64_t>& logical_sizes,
    const std::vector<int64_t>& logical_strides)
{
    TORCH_CHECK(tensor.numel() > 0, "Komodo-CANN V3 probe does not support empty tensors.");
    aclTensor* acl_tensor = aclCreateTensor(
        logical_sizes.data(),
        static_cast<uint64_t>(logical_sizes.size()),
        dtype,
        logical_strides.data(),
        0,
        ACL_FORMAT_ND,
        logical_sizes.data(),
        static_cast<uint64_t>(logical_sizes.size()),
        const_cast<void*>(tensor.const_data_ptr()));
    TORCH_CHECK(acl_tensor != nullptr, "aclCreateTensor failed in Komodo-CANN V3 probe.");
    return AclTensorHandle(acl_tensor);
}

AclTensorHandle make_acl_tensor(const at::Tensor& tensor)
{
    std::vector<int64_t> sizes = tensor.sizes().vec();
    return make_acl_tensor(tensor, to_acl_dtype(tensor.scalar_type()), sizes, contiguous_strides(sizes));
}

bool env_enabled(const char* name, bool default_value)
{
    const char* raw = std::getenv(name);
    if (raw == nullptr) {
        return default_value;
    }
    std::string value(raw);
    return !(value == "0" || value == "false" || value == "False" || value == "FALSE" || value == "off" ||
             value == "OFF" || value == "no" || value == "NO");
}

std::string normalized_env_value(const char* raw)
{
    std::string value(raw);
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), [](unsigned char ch) {
                    return !std::isspace(ch);
                }));
    value.erase(
        std::find_if(
            value.rbegin(),
            value.rend(),
            [](unsigned char ch) {
                return !std::isspace(ch);
            })
            .base(),
        value.end());
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

int inner_precise_value(int64_t rows, int64_t in_features, int64_t out_features, int64_t group_size)
{
    const char* raw = std::getenv(kInnerPreciseEnv);
    std::string value = raw == nullptr ? std::string("auto") : normalized_env_value(raw);
    if (value == "auto") {
        if (rows <= 16 && group_size == 32 && in_features >= 4096 && out_features >= in_features &&
            out_features <= in_features * 2) {
            return 1;
        }
        return 0;
    }
    char* end = nullptr;
    long parsed = std::strtol(value.c_str(), &end, 10);
    TORCH_CHECK(end != value.c_str() && *end == '\0', kInnerPreciseEnv, " must be 0, 1, or auto; got ", raw);
    TORCH_CHECK(parsed == 0 || parsed == 1, kInnerPreciseEnv, " must be 0 or 1; got ", parsed);
    return static_cast<int>(parsed);
}

int64_t infer_out_features(const at::Tensor& packed_weight, const at::Tensor& scales)
{
    if (scales.dim() > 0) {
        return scales.size(scales.dim() - 1);
    }
    return packed_weight.size(1) * 8;
}

void append_sizes(std::ostringstream& key, const at::Tensor& tensor)
{
    key << '[';
    for (const auto idx : c10::irange(tensor.dim())) {
        if (idx != 0) {
            key << ',';
        }
        key << tensor.size(static_cast<int64_t>(idx));
    }
    key << ']';
}

std::string v3_cache_key(
    const at::Tensor& x,
    const at::Tensor& packed_weight,
    const at::Tensor& scales,
    const at::Tensor& offsets,
    const c10::optional<at::Tensor>& bias,
    int64_t out_features,
    int64_t group_size)
{
    std::ostringstream key;
    key << "dev=" << x.device().index() << ";x=";
    append_sizes(key, x);
    key << ";wstore=";
    append_sizes(key, packed_weight);
    key << ";wlogical=[" << x.size(1) << ',' << out_features << ']';
    key << ";scale=";
    append_sizes(key, scales);
    key << ";offset=";
    append_sizes(key, offsets);
    key << ";bias=" << (bias.has_value() ? 1 : 0);
    if (bias.has_value()) {
        key << ':' << static_cast<int>(bias->scalar_type()) << ':';
        append_sizes(key, *bias);
    }
    key << ";xdtype=" << static_cast<int>(x.scalar_type());
    key << ";sdtype=" << static_cast<int>(scales.scalar_type());
    key << ";odtype=" << static_cast<int>(offsets.scalar_type());
    key << ";group=" << group_size;
    key << ";inner=" << inner_precise_value(x.size(0), x.size(1), out_features, group_size);
    return key.str();
}

struct V3ExecutorEntry {
    AclTensorHandle x_acl;
    AclTensorHandle weight_acl;
    AclTensorHandle scales_acl;
    AclTensorHandle offsets_acl;
    AclTensorHandle bias_acl;
    AclTensorHandle y_acl;
    aclOpExecutor* executor = nullptr;
    uint64_t workspace_size = 0;
    at::Tensor workspace;
};

using V3ExecutorCache = std::unordered_map<std::string, std::shared_ptr<V3ExecutorEntry>>;

V3ExecutorCache& executor_cache()
{
    // Keep cached ACL descriptors/executors alive for process lifetime. The
    // local CANN runtime owns executor cache bookkeeping; eager destruction
    // after launch double-freed during bring-up.
    static auto* cache = new V3ExecutorCache();
    return *cache;
}

std::mutex& executor_cache_mutex()
{
    static auto* mutex = new std::mutex();
    return *mutex;
}

void set_raw_tensor_addr(const AclTensorHandle& tensor, const at::Tensor& value, const char* name)
{
    aclnnStatus status = aclSetRawTensorAddr(tensor.get(), const_cast<void*>(value.const_data_ptr()));
    TORCH_CHECK(status == OK, "aclSetRawTensorAddr failed for ", name, " with status ", status);
}

void refresh_entry_addrs(
    const std::shared_ptr<V3ExecutorEntry>& entry,
    const at::Tensor& x,
    const at::Tensor& packed_weight,
    const at::Tensor& scales,
    const at::Tensor& offsets,
    const c10::optional<at::Tensor>& bias,
    const at::Tensor& y)
{
    set_raw_tensor_addr(entry->x_acl, x, "x");
    set_raw_tensor_addr(entry->weight_acl, packed_weight, "packed_weight");
    set_raw_tensor_addr(entry->scales_acl, scales, "scales");
    set_raw_tensor_addr(entry->offsets_acl, offsets, "offsets");
    if (bias.has_value()) {
        set_raw_tensor_addr(entry->bias_acl, *bias, "bias");
    }
    set_raw_tensor_addr(entry->y_acl, y, "y");
}

std::shared_ptr<V3ExecutorEntry> make_executor_entry(
    const at::Tensor& x,
    const at::Tensor& packed_weight,
    const at::Tensor& scales,
    const at::Tensor& offsets,
    const c10::optional<at::Tensor>& bias,
    const at::Tensor& y,
    int64_t out_features,
    int64_t group_size)
{
    auto entry = std::make_shared<V3ExecutorEntry>();
    std::vector<int64_t> weight_sizes = {x.size(1), out_features};
    entry->x_acl = make_acl_tensor(x);
    entry->weight_acl = make_acl_tensor(packed_weight, ACL_INT4, weight_sizes, contiguous_strides(weight_sizes));
    entry->scales_acl = make_acl_tensor(scales);
    entry->offsets_acl = make_acl_tensor(offsets);
    entry->bias_acl = bias.has_value() ? make_acl_tensor(*bias) : AclTensorHandle();
    entry->y_acl = make_acl_tensor(y);

    int inner_precise = inner_precise_value(x.size(0), x.size(1), out_features, group_size);
    aclnnStatus status = aclnnWeightQuantBatchMatmulV3GetWorkspaceSize(
        entry->x_acl.get(),
        entry->weight_acl.get(),
        entry->scales_acl.get(),
        entry->offsets_acl.get(),
        nullptr,
        nullptr,
        entry->bias_acl.get(),
        static_cast<int>(group_size),
        inner_precise,
        entry->y_acl.get(),
        &entry->workspace_size,
        &entry->executor);
    TORCH_CHECK(status == OK, "aclnnWeightQuantBatchMatmulV3GetWorkspaceSize failed with status ", status);
    TORCH_CHECK(entry->executor != nullptr, "aclnnWeightQuantBatchMatmulV3GetWorkspaceSize returned a null executor.");

    status = aclSetAclOpExecutorRepeatable(entry->executor);
    TORCH_CHECK(status == OK, "aclSetAclOpExecutorRepeatable failed with status ", status);
    return entry;
}

std::shared_ptr<V3ExecutorEntry> get_or_create_executor_entry(
    const std::string& key,
    const at::Tensor& x,
    const at::Tensor& packed_weight,
    const at::Tensor& scales,
    const at::Tensor& offsets,
    const c10::optional<at::Tensor>& bias,
    const at::Tensor& y,
    int64_t out_features,
    int64_t group_size)
{
    std::lock_guard<std::mutex> lock(executor_cache_mutex());
    auto& cache = executor_cache();
    auto it = cache.find(key);
    if (it != cache.end()) {
        refresh_entry_addrs(it->second, x, packed_weight, scales, offsets, bias, y);
        return it->second;
    }
    auto entry = make_executor_entry(x, packed_weight, scales, offsets, bias, y, out_features, group_size);
    cache.emplace(key, entry);
    return entry;
}

void* workspace_ptr_for_entry(const std::shared_ptr<V3ExecutorEntry>& entry, const at::Tensor& x)
{
    if (entry->workspace_size == 0) {
        return nullptr;
    }
    if (!env_enabled(kWorkspaceCacheEnv, true)) {
        entry->workspace = at::empty({static_cast<int64_t>(entry->workspace_size)}, x.options().dtype(at::kByte));
        return entry->workspace.data_ptr();
    }
    if (!entry->workspace.defined() ||
        entry->workspace.numel() < static_cast<int64_t>(entry->workspace_size) ||
        entry->workspace.device() != x.device()) {
        entry->workspace = at::empty({static_cast<int64_t>(entry->workspace_size)}, x.options().dtype(at::kByte));
    }
    return entry->workspace.data_ptr();
}

at::Tensor execute_v3(
    const at::Tensor& x,
    const at::Tensor& packed_weight,
    const at::Tensor& scales,
    const at::Tensor& offsets,
    const c10::optional<at::Tensor>& bias,
    const at::Tensor& y,
    int64_t out_features,
    int64_t group_size)
{
    std::shared_ptr<V3ExecutorEntry> entry;
    if (env_enabled(kExecutorCacheEnv, true)) {
        entry = get_or_create_executor_entry(
            v3_cache_key(x, packed_weight, scales, offsets, bias, out_features, group_size),
            x,
            packed_weight,
            scales,
            offsets,
            bias,
            y,
            out_features,
            group_size);
    } else {
        entry = make_executor_entry(x, packed_weight, scales, offsets, bias, y, out_features, group_size);
    }

    void* workspace_ptr = workspace_ptr_for_entry(entry, x);
    aclrtStream stream = c10_npu::getCurrentNPUStream(x.device().index()).stream(false);
    aclnnStatus status = aclnnWeightQuantBatchMatmulV3(workspace_ptr, entry->workspace_size, entry->executor, stream);
    TORCH_CHECK(status == OK, "aclnnWeightQuantBatchMatmulV3 failed with status ", status);
    return y;
}

at::Tensor w4a16_matmul_v3(
    const at::Tensor& x,
    const at::Tensor& packed_weight,
    const at::Tensor& scales,
    const at::Tensor& offsets,
    const c10::optional<at::Tensor>& bias,
    int64_t group_size,
    int64_t split_k,
    int64_t base_m,
    int64_t base_n,
    int64_t base_k)
{
    (void)split_k;
    (void)base_m;
    (void)base_n;
    (void)base_k;

    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "Komodo-CANN V3 probe expects x on NPU.");
    TORCH_CHECK(
        packed_weight.device() == x.device() && scales.device() == x.device() && offsets.device() == x.device(),
        "Komodo-CANN V3 probe expects x, packed_weight, scales, and offsets on the same NPU.");
    TORCH_CHECK(x.scalar_type() == at::kHalf, "Komodo-CANN V3 probe expects FP16 activations.");
    TORCH_CHECK(scales.scalar_type() == at::kHalf, "Komodo-CANN V3 probe expects FP16 antiquant scales.");
    TORCH_CHECK(offsets.scalar_type() == at::kHalf, "Komodo-CANN V3 probe expects FP16 antiquant offsets.");
    TORCH_CHECK(x.dim() == 2, "Komodo-CANN V3 probe currently expects flat 2D activations.");
    TORCH_CHECK(packed_weight.dim() == 2, "Komodo-CANN V3 probe expects a 2D int4-packed weight.");

    c10_npu::OptionalNPUGuard guard(x.device());

    at::Tensor x_arg = x.is_contiguous() ? x : x.contiguous();
    at::Tensor weight_arg = packed_weight.is_contiguous() ? packed_weight : packed_weight.contiguous();
    at::Tensor scales_arg = scales.is_contiguous() ? scales : scales.contiguous();
    at::Tensor offsets_arg = offsets.is_contiguous() ? offsets : offsets.contiguous();
    c10::optional<at::Tensor> bias_arg = bias;
    if (bias_arg.has_value()) {
        TORCH_CHECK(bias_arg->device() == x.device(), "Komodo-CANN V3 probe expects bias on the same NPU.");
        TORCH_CHECK(
            bias_arg->scalar_type() == at::kHalf || bias_arg->scalar_type() == at::kFloat,
            "Komodo-CANN V3 probe expects FP16 or FP32 bias.");
        if (!bias_arg->is_contiguous()) {
            bias_arg = bias_arg->contiguous();
        }
    }

    std::vector<int64_t> out_sizes = x_arg.sizes().vec();
    out_sizes.back() = infer_out_features(weight_arg, scales_arg);
    at::Tensor y = at::empty(out_sizes, x_arg.options());

    return execute_v3(x_arg, weight_arg, scales_arg, offsets_arg, bias_arg, y, out_sizes.back(), group_size);
}

}  // namespace

TORCH_LIBRARY(gptqmodel_komodo_cann, m)
{
    m.def(
        "w4a16_matmul(Tensor x, Tensor packed_weight, Tensor scales, Tensor offsets, Tensor? bias, "
        "int group_size, int split_k, int base_m, int base_n, int base_k) -> Tensor");
}

TORCH_LIBRARY_IMPL(gptqmodel_komodo_cann, PrivateUse1, m)
{
    m.impl("w4a16_matmul", &w4a16_matmul_v3);
}
