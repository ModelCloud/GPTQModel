// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include <torch/extension.h>

#include <c10/core/DeviceType.h>
#include <c10/util/Optional.h>
#include <dlfcn.h>
#include <cstdlib>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "acl/acl_base.h"
#include "aclnn/acl_meta.h"
#include "torch_npu/csrc/framework/OpCommand.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace {

constexpr const char* kCustomOpApiLibEnv = "GPTQMODEL_KOMODO_CANN_ASCENDC_OPAPI_LIB";
constexpr const char* kCustomOpApiLibName = "libcust_opapi.so";

using GetWorkspaceFn = aclnnStatus (*)(
    const aclTensor* x,
    const aclTensor* packedWeight,
    const aclTensor* scales,
    const aclTensor* offsets,
    const aclTensor* biasOptional,
    int64_t groupSize,
    int64_t splitK,
    int64_t baseM,
    int64_t baseN,
    int64_t baseK,
    const aclTensor* out,
    uint64_t* workspaceSize,
    aclOpExecutor** executor);

using RunFn = aclnnStatus (*)(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

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

struct AscendcApi {
    void* handle = nullptr;
    GetWorkspaceFn get_workspace = nullptr;
    RunFn run = nullptr;
    std::string error;
};

std::mutex& api_mutex()
{
    static auto* mutex = new std::mutex();
    return *mutex;
}

AscendcApi& api_state()
{
    static auto* state = new AscendcApi();
    return *state;
}

std::string dl_error()
{
    const char* raw = dlerror();
    return raw == nullptr ? std::string("unknown dlerror") : std::string(raw);
}

void* checked_dlsym(void* handle, const char* symbol)
{
    dlerror();
    void* ptr = dlsym(handle, symbol);
    const char* err = dlerror();
    TORCH_CHECK(ptr != nullptr && err == nullptr, "Failed to resolve ", symbol, " from ", kCustomOpApiLibName, ": ",
                err == nullptr ? "symbol is null" : err);
    return ptr;
}

AscendcApi& load_api()
{
    std::lock_guard<std::mutex> lock(api_mutex());
    AscendcApi& state = api_state();
    if (state.handle != nullptr) {
        return state;
    }

    const char* explicit_lib = std::getenv(kCustomOpApiLibEnv);
    const char* lib = explicit_lib != nullptr && explicit_lib[0] != '\0' ? explicit_lib : kCustomOpApiLibName;
    dlerror();
    state.handle = dlopen(lib, RTLD_NOW | RTLD_LOCAL);
    if (state.handle == nullptr) {
        state.error = dl_error();
        TORCH_CHECK(false, "Failed to load Komodo-CANN Ascend C op API library `", lib, "`: ", state.error,
                    ". Source the custom OPP set_env.bash or set ", kCustomOpApiLibEnv, ".");
    }

    state.get_workspace =
        reinterpret_cast<GetWorkspaceFn>(checked_dlsym(state.handle, "aclnnKomodoCannW4A16MatmulGetWorkspaceSize"));
    state.run = reinterpret_cast<RunFn>(checked_dlsym(state.handle, "aclnnKomodoCannW4A16Matmul"));
    return state;
}

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
            TORCH_CHECK(false, "Unsupported dtype for Komodo-CANN Ascend C bridge: ", dtype);
    }
}

AclTensorHandle make_acl_tensor(
    const at::Tensor& tensor,
    aclDataType dtype,
    const std::vector<int64_t>& logical_sizes,
    const std::vector<int64_t>& logical_strides)
{
    TORCH_CHECK(tensor.numel() > 0, "Komodo-CANN Ascend C bridge does not support empty tensors.");
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
    TORCH_CHECK(acl_tensor != nullptr, "aclCreateTensor failed in Komodo-CANN Ascend C bridge.");
    return AclTensorHandle(acl_tensor);
}

AclTensorHandle make_acl_tensor(const at::Tensor& tensor)
{
    std::vector<int64_t> sizes = tensor.sizes().vec();
    return make_acl_tensor(tensor, to_acl_dtype(tensor.scalar_type()), sizes, contiguous_strides(sizes));
}

int64_t infer_out_features(const at::Tensor& packed_weight, const at::Tensor& scales)
{
    if (scales.dim() > 0) {
        return scales.size(scales.dim() - 1);
    }
    return packed_weight.size(1) * 8;
}

void check_inputs(
    const at::Tensor& x,
    const at::Tensor& packed_weight,
    const at::Tensor& scales,
    const at::Tensor& offsets,
    const c10::optional<at::Tensor>& bias,
    int64_t group_size,
    int64_t out_features)
{
    TORCH_CHECK(x.device().type() == c10::DeviceType::PrivateUse1, "Komodo-CANN Ascend C bridge expects x on NPU.");
    TORCH_CHECK(
        packed_weight.device() == x.device() && scales.device() == x.device() && offsets.device() == x.device(),
        "Komodo-CANN Ascend C bridge expects x, packed_weight, scales, and offsets on the same NPU.");
    TORCH_CHECK(x.scalar_type() == at::kHalf, "Komodo-CANN Ascend C bridge expects FP16 activations.");
    TORCH_CHECK(packed_weight.scalar_type() == at::kInt, "Komodo-CANN Ascend C bridge expects INT32 packed weights.");
    TORCH_CHECK(scales.scalar_type() == at::kHalf, "Komodo-CANN Ascend C bridge expects FP16 scales.");
    TORCH_CHECK(offsets.scalar_type() == at::kHalf, "Komodo-CANN Ascend C bridge expects FP16 offsets.");
    TORCH_CHECK(x.dim() == 2, "Komodo-CANN Ascend C bridge expects flat 2D activations.");
    TORCH_CHECK(packed_weight.dim() == 2, "Komodo-CANN Ascend C bridge expects packed_weight [K, N / 8].");
    TORCH_CHECK(scales.dim() == 2 && offsets.dim() == 2, "Komodo-CANN Ascend C bridge expects 2D scales/offsets.");
    TORCH_CHECK(packed_weight.size(0) == x.size(1), "Komodo-CANN Ascend C packed K must match activation K.");
    TORCH_CHECK(packed_weight.size(1) * 8 == out_features, "Komodo-CANN Ascend C packed N/8 must match scales N.");
    TORCH_CHECK(offsets.sizes() == scales.sizes(), "Komodo-CANN Ascend C scales and offsets must have same shape.");
    TORCH_CHECK(group_size == 0 || group_size >= 32, "Komodo-CANN Ascend C supports group_size 0 or >= 32.");
    TORCH_CHECK(group_size == 0 || x.size(1) % group_size == 0, "Komodo-CANN Ascend C K must divide group_size.");
    const int64_t expected_groups = group_size == 0 ? 1 : x.size(1) / group_size;
    TORCH_CHECK(scales.size(0) == expected_groups, "Komodo-CANN Ascend C scales group count mismatch.");
    if (bias.has_value()) {
        TORCH_CHECK(bias->device() == x.device(), "Komodo-CANN Ascend C bridge expects bias on the same NPU.");
        TORCH_CHECK(bias->scalar_type() == at::kHalf, "Komodo-CANN Ascend C bridge expects FP16 bias.");
        TORCH_CHECK(bias->dim() == 1 && bias->size(0) == out_features, "Komodo-CANN Ascend C bias must be [N].");
    }
}

at::Tensor komodo_cann_w4_a16_matmul(
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
    const int64_t out_features = infer_out_features(packed_weight, scales);
    check_inputs(x, packed_weight, scales, offsets, bias, group_size, out_features);
    c10_npu::OptionalNPUGuard guard(x.device());

    at::Tensor x_arg = x.is_contiguous() ? x : x.contiguous();
    at::Tensor weight_arg = packed_weight.is_contiguous() ? packed_weight : packed_weight.contiguous();
    at::Tensor scales_arg = scales.is_contiguous() ? scales : scales.contiguous();
    at::Tensor offsets_arg = offsets.is_contiguous() ? offsets : offsets.contiguous();
    c10::optional<at::Tensor> bias_arg = bias;
    if (bias_arg.has_value() && !bias_arg->is_contiguous()) {
        bias_arg = bias_arg->contiguous();
    }

    at::Tensor y = at::empty({x_arg.size(0), out_features}, x_arg.options());
    AclTensorHandle x_acl = make_acl_tensor(x_arg);
    AclTensorHandle weight_acl = make_acl_tensor(weight_arg);
    AclTensorHandle scales_acl = make_acl_tensor(scales_arg);
    AclTensorHandle offsets_acl = make_acl_tensor(offsets_arg);
    AclTensorHandle bias_acl = bias_arg.has_value() ? make_acl_tensor(*bias_arg) : AclTensorHandle();
    AclTensorHandle y_acl = make_acl_tensor(y);

    AscendcApi& api = load_api();
    uint64_t workspace_size = 0;
    aclOpExecutor* executor = nullptr;
    aclnnStatus status = api.get_workspace(
        x_acl.get(),
        weight_acl.get(),
        scales_acl.get(),
        offsets_acl.get(),
        bias_acl.get(),
        group_size,
        split_k,
        base_m,
        base_n,
        base_k,
        y_acl.get(),
        &workspace_size,
        &executor);
    TORCH_CHECK(status == OK, "aclnnKomodoCannW4A16MatmulGetWorkspaceSize failed with status ", status);
    TORCH_CHECK(executor != nullptr, "aclnnKomodoCannW4A16MatmulGetWorkspaceSize returned a null executor.");

    at::Tensor workspace;
    void* workspace_ptr = nullptr;
    if (workspace_size > 0) {
        workspace = at::empty({static_cast<int64_t>(workspace_size)}, x_arg.options().dtype(at::kByte));
        workspace_ptr = workspace.data_ptr();
    }

    aclrtStream stream = c10_npu::getCurrentNPUStream(x.device().index()).stream(false);
    auto acl_call = [&api, workspace_ptr, workspace_size, executor, stream]() -> int {
        const aclnnStatus run_status = api.run(workspace_ptr, workspace_size, executor, stream);
        TORCH_CHECK(run_status == OK, "aclnnKomodoCannW4A16Matmul failed with status ", run_status);
        return static_cast<int>(run_status);
    };
    at_npu::native::OpCommand::RunOpApiV2("aclnnKomodoCannW4A16Matmul", acl_call);
    return y;
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(gptqmodel_komodo_cann, m)
{
    m.def(
        "komodo_cann_w4_a16_matmul(Tensor x, Tensor packed_weight, Tensor scales, Tensor offsets, Tensor? bias, "
        "int group_size, int split_k, int base_m, int base_n, int base_k) -> Tensor");
}

TORCH_LIBRARY_IMPL(gptqmodel_komodo_cann, PrivateUse1, m)
{
    m.impl("komodo_cann_w4_a16_matmul", &komodo_cann_w4_a16_matmul);
}
