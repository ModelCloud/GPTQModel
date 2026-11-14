#include "./../torch_stable_compat.h"
#include "ext_common.h"
#include <optional>

#include "cuda/q_matrix.cuh"
#include "cuda/q_gemm.cuh"
#include "config.h"
#include "q_matrix_clone.h"

int64_t make_q_matrix(
    torch::Tensor q_weight,
    std::optional<torch::Tensor> q_perm_opt,
    std::optional<torch::Tensor> q_invperm_opt,
    std::optional<torch::Tensor> q_scale_opt,
    std::optional<torch::Tensor> q_scale_max_opt,
    std::optional<torch::Tensor> q_groups_opt,
    std::optional<torch::Tensor> gptq_qzeros_opt,
    std::optional<torch::Tensor> gptq_scales_opt,
    std::optional<torch::Tensor> gptq_g_idx_opt,
    bool already_shuffled
)
{
    torch::Tensor q_perm = q_perm_opt.value_or(torch::Tensor());
    torch::Tensor q_invperm = q_invperm_opt.value_or(torch::Tensor());
    torch::Tensor q_scale = q_scale_opt.value_or(torch::Tensor());
    torch::Tensor q_scale_max = q_scale_max_opt.value_or(torch::Tensor());
    torch::Tensor q_groups = q_groups_opt.value_or(torch::Tensor());
    torch::Tensor gptq_qzeros = gptq_qzeros_opt.value_or(torch::Tensor());
    torch::Tensor gptq_scales = gptq_scales_opt.value_or(torch::Tensor());
    torch::Tensor gptq_g_idx = gptq_g_idx_opt.value_or(torch::Tensor());

    TORCH_CHECK_DTYPE(q_weight, kInt);
    TORCH_CHECK(!q_perm.defined() || at::is_meta(q_perm.device()) || q_perm.scalar_type() == torch::kShort,
                "q_perm is incorrect datatype, must be kShort");
    TORCH_CHECK(!q_invperm.defined() || at::is_meta(q_invperm.device()) || q_invperm.scalar_type() == torch::kShort,
                "q_invperm is incorrect datatype, must be kShort");
    TORCH_CHECK(!q_scale.defined() || at::is_meta(q_scale.device()) || q_scale.scalar_type() == torch::kInt,
                "q_scale is incorrect datatype, must be kInt");
    TORCH_CHECK(!q_scale_max.defined() || at::is_meta(q_scale_max.device()) || q_scale_max.scalar_type() == torch::kHalf,
                "q_scale_max is incorrect datatype, must be kHalf");
    TORCH_CHECK(!q_groups.defined() || at::is_meta(q_groups.device()) || q_groups.scalar_type() == torch::kShort,
                "q_groups is incorrect datatype, must be kShort");
    TORCH_CHECK(!gptq_qzeros.defined() || at::is_meta(gptq_qzeros.device()) || gptq_qzeros.scalar_type() == torch::kInt,
                "gptq_qzeros is incorrect datatype, must be kInt");
    TORCH_CHECK(!gptq_scales.defined() || at::is_meta(gptq_scales.device()) || gptq_scales.scalar_type() == torch::kHalf,
                "gptq_scales is incorrect datatype, must be kHalf");
    TORCH_CHECK(!gptq_g_idx.defined() || at::is_meta(gptq_g_idx.device()) || gptq_g_idx.scalar_type() == torch::kInt,
                "gptq_g_idx is incorrect datatype, must be kInt");

    TORCH_CHECK(!q_perm.defined() || !q_invperm.defined() || q_perm.size(0) == q_invperm.size(0),
                "q_perm and q_invperm have incompatible shapes");

    int device = q_weight.device().index();
    int width = q_weight.size(1);
    int groups;
    int height;

    if (q_scale.defined() && !at::is_meta(q_scale.device()))
    {
        TORCH_CHECK_SHAPES(q_weight, 1, q_scale, 1, 8);
        TORCH_CHECK_SHAPES(q_scale_max, 0, q_scale, 0, 1);
        groups = q_scale.size(0);
        height = q_invperm.size(0);
    }
    else
    {
        TORCH_CHECK_SHAPES(q_weight, 1, gptq_qzeros, 1, 8);
        TORCH_CHECK_SHAPES(q_weight, 1, gptq_scales, 1, 1);
        groups = gptq_qzeros.size(0);
        height = q_weight.size(0) * 8;
    }

    // No dense reconstruction buffer is attached: the quantized kernel above
    // decodes inside its launch, and the dense fallback allocates a transient
    // tensor per call in gemm_half_q_half().

    QMatrix* m = new QMatrix(
        device,
        height,
        width,
        groups,
        (uint32_t*) q_weight.data_ptr(),
        (!q_perm.defined() || at::is_meta(q_perm.device())) ? NULL : (uint16_t*) q_perm.data_ptr(),
        (!q_invperm.defined() || at::is_meta(q_invperm.device())) ? NULL : (uint16_t*) q_invperm.data_ptr(),
        (!q_scale.defined() || at::is_meta(q_scale.device())) ? NULL : (uint32_t*) q_scale.data_ptr(),
        (!q_scale_max.defined() || at::is_meta(q_scale_max.device())) ? NULL : (half*) q_scale_max.data_ptr(),
        (!q_groups.defined() || at::is_meta(q_groups.device())) ? NULL : (uint16_t*) q_groups.data_ptr(),
        (!gptq_qzeros.defined() || at::is_meta(gptq_qzeros.device())) ? NULL : (uint32_t*) gptq_qzeros.data_ptr(),
        (!gptq_scales.defined() || at::is_meta(gptq_scales.device())) ? NULL : (half*) gptq_scales.data_ptr(),
        (!gptq_g_idx.defined() || at::is_meta(gptq_g_idx.device())) ? NULL : (uint32_t*) gptq_g_idx.data_ptr(),
        already_shuffled
    );

    return static_cast<int64_t>(reinterpret_cast<uintptr_t>(m));
}

void gemm_half_q_half(
    torch::Tensor a,
    int64_t b,
    torch::Tensor& c,
    bool force_cuda
)
{
    QMatrix* qm = reinterpret_cast<QMatrix*>(static_cast<uintptr_t>(b));

    TORCH_CHECK(a.scalar_type() == at::kHalf || a.scalar_type() == at::kBFloat16 || a.scalar_type() == at::kFloat,
                "ExLlamaV2 requires FP16, BF16 or FP32 activations");
    TORCH_CHECK(c.scalar_type() == a.scalar_type(), "output dtype must match input");
    TORCH_CHECK_SHAPES(a, 0, c, 0, 1);
    TORCH_CHECK(qm->height == a.size(1), "a and b have incompatible shapes");
    TORCH_CHECK(qm->width == c.size(1), "b and c have incompatible shapes");
    const at::cuda::OptionalCUDAGuard device_guard(device_of(a));

    if (a.scalar_type() != at::kHalf || (a.size(0) > MAX_Q_GEMM_ROWS && !force_cuda)) {
        auto weights = torch::empty({qm->height, qm->width}, at::TensorOptions().dtype(at::kHalf).device(a.device()));
        qm->reconstruct(reinterpret_cast<half*>(weights.data_ptr()));
        if (a.scalar_type() == at::kHalf) {
            gemm_half_half_cuda(
                at::cuda::getCurrentCUDABlasHandle(),
                (const half*) a.data_ptr(),
                (const half*) weights.data_ptr(),
                (half*) c.data_ptr(),
                c.size(0),
                c.size(1),
                a.size(1),
                true
            );
        } else {
            torch::stable::copy_(c, torch::stable::matmul(at::to(a, at::kFloat), at::to(weights, at::kFloat)));
        }
        return;
    }

    gemm_half_q_half_cuda(
        at::cuda::getCurrentCUDABlasHandle(),
        (const half*) a.data_ptr(),
        qm,
        (half*) c.data_ptr(),
        c.size(0),
        c.size(1),
        a.size(1),
        true
    );
}

STABLE_TORCH_LIBRARY(gptqmodel_exllamav2_awq, m)
{
    m.def("make_q_matrix_awq(Tensor q_weight, Tensor? q_perm, Tensor? q_invperm, Tensor? q_scale, Tensor? q_scale_max, Tensor? q_groups, Tensor? gptq_qzeros, Tensor? gptq_scales, Tensor? gptq_g_idx, bool already_shuffled=False) -> int");
    m.def("clone_q_matrix_awq(int handle, Tensor weight, Tensor zeros, Tensor scales, Tensor? perm=None, Tensor? invperm=None) -> int");
    m.def("free_q_matrix_awq(int handle) -> ()");
    m.def("gemm_half_q_half_awq(Tensor a, int b, Tensor(a!) c, bool force_cuda=False) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(gptqmodel_exllamav2_awq, CUDA, m)
{
    m.impl("make_q_matrix_awq", TORCH_BOX(&make_q_matrix));
    m.impl("clone_q_matrix_awq", TORCH_BOX(&clone_q_matrix));
    m.impl("gemm_half_q_half_awq", TORCH_BOX(&gemm_half_q_half));
}

STABLE_TORCH_LIBRARY_IMPL(gptqmodel_exllamav2_awq, CompositeExplicitAutograd, m)
{
    m.impl("free_q_matrix_awq", TORCH_BOX(&free_q_matrix));
}
