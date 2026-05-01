#include "komodo_cann_w4_a16_matmul_tiling.h"
#include "komodo_cann_w4_a16_matmul_tiling_key.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
constexpr size_t kInputX = 0;
constexpr size_t kInputPackedWeight = 1;
constexpr size_t kInputScales = 2;
constexpr size_t kInputOffsets = 3;
constexpr size_t kInputBias = 4;

constexpr size_t kAttrGroupSize = 0;
constexpr size_t kAttrSplitK = 1;
constexpr size_t kAttrBaseM = 2;
constexpr size_t kAttrBaseN = 3;
constexpr size_t kAttrBaseK = 4;

constexpr uint32_t kKernelModeScalar = 0;
constexpr uint32_t kKernelModeStagedDequant = 1;
constexpr uint32_t kStagingSlots = 2;
constexpr uint64_t kStagingAlignmentBytes = 512;
// 910B CANN reserves this system workspace before the user workspace returned by GetUserWorkspace().
constexpr uint64_t kCubeSysWorkspaceBytes = 16ULL * 1024ULL * 1024ULL;

uint32_t AttrAsU32(const gert::RuntimeAttrs* attrs, size_t index, uint32_t fallback)
{
    if (attrs == nullptr) {
        return fallback;
    }
    const int64_t* value = attrs->GetInt(index);
    if (value == nullptr || *value < 0) {
        return fallback;
    }
    return static_cast<uint32_t>(*value);
}

uint32_t AttrAbsAsU32(const gert::RuntimeAttrs* attrs, size_t index, uint32_t fallback)
{
    if (attrs == nullptr) {
        return fallback;
    }
    const int64_t* value = attrs->GetInt(index);
    if (value == nullptr || *value == 0) {
        return fallback;
    }
    const uint64_t magnitude = *value < 0 ? static_cast<uint64_t>(-(*value + 1)) + 1U : static_cast<uint64_t>(*value);
    constexpr uint64_t kMaxU32 = static_cast<uint64_t>(0xffffffffU);
    return static_cast<uint32_t>(magnitude > kMaxU32 ? kMaxU32 : magnitude);
}

uint32_t AttrIsNegative(const gert::RuntimeAttrs* attrs, size_t index)
{
    if (attrs == nullptr) {
        return 0;
    }
    const int64_t* value = attrs->GetInt(index);
    return value != nullptr && *value < 0 ? 1U : 0U;
}

uint32_t ClampU64ToU32(uint64_t value)
{
    constexpr uint64_t kMaxU32 = static_cast<uint64_t>(0xffffffffU);
    return static_cast<uint32_t>(value > kMaxU32 ? kMaxU32 : value);
}

uint64_t AlignUpU64(uint64_t value, uint64_t alignment)
{
    if (alignment == 0) {
        return value;
    }
    return ((value + alignment - 1) / alignment) * alignment;
}

uint32_t CeilDivU32(uint32_t value, uint32_t divisor)
{
    if (divisor == 0) {
        return 0;
    }
    return (value + divisor - 1) / divisor;
}

uint32_t PickBlockDim(uint32_t packed_words, uint32_t aiv_cores)
{
    if (packed_words <= 1 || aiv_cores == 0) {
        return 1;
    }
    uint32_t block_dim = packed_words < aiv_cores ? packed_words : aiv_cores;
    return block_dim < 8 ? block_dim : 8;
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    const gert::StorageShape* x_shape = context->GetInputShape(kInputX);
    const gert::StorageShape* packed_shape = context->GetInputShape(kInputPackedWeight);
    const gert::StorageShape* scales_shape = context->GetInputShape(kInputScales);
    const gert::StorageShape* offsets_shape = context->GetInputShape(kInputOffsets);
    if (x_shape == nullptr || packed_shape == nullptr || scales_shape == nullptr || offsets_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    const auto& x = x_shape->GetStorageShape();
    const auto& packed = packed_shape->GetStorageShape();
    const auto& scales = scales_shape->GetStorageShape();
    const auto& offsets = offsets_shape->GetStorageShape();
    if (x.GetDimNum() != 2 || packed.GetDimNum() != 2 || scales.GetDimNum() != 2 || offsets.GetDimNum() != 2) {
        return ge::GRAPH_FAILED;
    }

    const int64_t rows64 = x.GetDim(0);
    const int64_t k64 = x.GetDim(1);
    const int64_t packed_k64 = packed.GetDim(0);
    const int64_t packed_n_words64 = packed.GetDim(1);
    const int64_t scale_groups64 = scales.GetDim(0);
    const int64_t n64 = scales.GetDim(1);
    if (rows64 <= 0 || k64 <= 0 || packed_k64 != k64 || packed_n_words64 <= 0 || n64 <= 0 ||
        offsets.GetDim(0) != scale_groups64 || offsets.GetDim(1) != n64 || packed_n_words64 * 8 != n64) {
        return ge::GRAPH_FAILED;
    }

    const auto attrs = context->GetAttrs();
    const uint32_t group_size = AttrAsU32(attrs, kAttrGroupSize, 0);
    if (group_size != 0 && (group_size < 32 || (static_cast<uint64_t>(k64) % group_size) != 0)) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t expected_groups =
        group_size == 0 ? 1 : static_cast<uint32_t>((static_cast<uint64_t>(k64) + group_size - 1) / group_size);
    if (static_cast<uint32_t>(scale_groups64) != expected_groups) {
        return ge::GRAPH_FAILED;
    }

    const uint64_t total_outputs64 = static_cast<uint64_t>(rows64) * static_cast<uint64_t>(n64);
    if (total_outputs64 == 0 || total_outputs64 > static_cast<uint64_t>(0xffffffffU)) {
        return ge::GRAPH_FAILED;
    }
    const uint32_t total_outputs = static_cast<uint32_t>(total_outputs64);

    KomodoCannW4A16MatmulTilingData tiling;
    tiling.set_rows(static_cast<uint32_t>(rows64));
    tiling.set_in_features(static_cast<uint32_t>(k64));
    tiling.set_out_features(static_cast<uint32_t>(n64));
    tiling.set_group_size(group_size);
    tiling.set_has_bias(context->GetOptionalInputShape(kInputBias) != nullptr ? 1U : 0U);
    tiling.set_zero_offsets(AttrIsNegative(attrs, kAttrBaseK));
    const uint32_t cube_workspace_requested = AttrIsNegative(attrs, kAttrSplitK);
    const uint32_t requested_split_k = AttrAbsAsU32(attrs, kAttrSplitK, 1);
    tiling.set_split_k(requested_split_k);
    tiling.set_base_m(AttrAsU32(attrs, kAttrBaseM, rows64 <= 16 ? 16 : 128));
    const uint32_t requested_base_n = AttrAbsAsU32(attrs, kAttrBaseN, 256);
    tiling.set_base_n(requested_base_n);
    const uint32_t requested_base_k = AttrAbsAsU32(attrs, kAttrBaseK, 64);
    tiling.set_base_k(requested_base_k);
    tiling.set_total_outputs(total_outputs);

    uint64_t ub_size = 0;
    uint64_t l1_size = 0;
    uint64_t l0a_size = 0;
    uint64_t l0b_size = 0;
    uint64_t l0c_size = 0;
    uint32_t aiv_cores = 1;
    auto platform_info = context->GetPlatformInfo();
    if (platform_info != nullptr) {
        auto platform = platform_ascendc::PlatformAscendC(platform_info);
        aiv_cores = platform.GetCoreNumAiv();
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1_size);
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_A, l0a_size);
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_B, l0b_size);
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0c_size);
    }
    tiling.set_ub_bytes(ClampU64ToU32(ub_size));
    tiling.set_l1_bytes(ClampU64ToU32(l1_size));
    tiling.set_l0a_bytes(ClampU64ToU32(l0a_size));
    tiling.set_l0b_bytes(ClampU64ToU32(l0b_size));
    tiling.set_l0c_bytes(ClampU64ToU32(l0c_size));

    const uint32_t block_dim = PickBlockDim(static_cast<uint32_t>(packed_n_words64), aiv_cores);
    tiling.set_block_dim(block_dim);
    tiling.set_kernel_mode(kKernelModeScalar);
    tiling.set_staging_blocks(0);
    tiling.set_staging_slots(0);
    tiling.set_staging_tile_bytes(0);
    tiling.set_staging_workspace_bytes(0);
    tiling.set_staging_workspace_offset(0);
    tiling.set_cube_workspace_bytes(0);
    uint64_t tiling_key = GET_TPL_TILING_KEY(kKomodoCannLaunchModeAiv);

    if (AttrIsNegative(attrs, kAttrBaseN) != 0 && requested_base_n != 0 && requested_base_k != 0) {
        const uint32_t n_tiles = CeilDivU32(static_cast<uint32_t>(n64), requested_base_n);
        const uint32_t split_k = requested_split_k == 0 ? 1 : requested_split_k;
        const uint64_t planned_stage_blocks = static_cast<uint64_t>(n_tiles) * static_cast<uint64_t>(split_k);
        uint32_t staging_blocks =
            planned_stage_blocks < block_dim ? static_cast<uint32_t>(planned_stage_blocks) : block_dim;
        if (staging_blocks == 0) {
            staging_blocks = 1;
        }
        const uint64_t tile_bytes = AlignUpU64(
            static_cast<uint64_t>(requested_base_k) * static_cast<uint64_t>(requested_base_n) * sizeof(uint16_t),
            kStagingAlignmentBytes);
        const uint64_t staging_workspace_bytes = tile_bytes * static_cast<uint64_t>(kStagingSlots) *
            static_cast<uint64_t>(staging_blocks);
        const uint64_t cube_workspace_bytes = cube_workspace_requested != 0 ? kCubeSysWorkspaceBytes : 0;
        const uint64_t workspace_bytes = cube_workspace_bytes + staging_workspace_bytes;
        const uint64_t dense_dequant_bytes =
            static_cast<uint64_t>(k64) * static_cast<uint64_t>(n64) * sizeof(uint16_t);
        if (staging_workspace_bytes > 0 && staging_workspace_bytes < dense_dequant_bytes) {
            tiling.set_kernel_mode(kKernelModeStagedDequant);
            tiling.set_staging_blocks(staging_blocks);
            tiling.set_staging_slots(kStagingSlots);
            tiling.set_staging_tile_bytes(ClampU64ToU32(tile_bytes));
            tiling.set_staging_workspace_bytes(ClampU64ToU32(staging_workspace_bytes));
            tiling.set_staging_workspace_offset(0);
            tiling.set_cube_workspace_bytes(ClampU64ToU32(cube_workspace_bytes));
#ifdef KOMODO_CANN_EXPERIMENTAL_MIXED_LAUNCH
            tiling_key = GET_TPL_TILING_KEY(kKomodoCannLaunchModeMixedAicAiv);
#endif
            size_t* workspaces = context->GetWorkspaceSizes(1);
            if (workspaces == nullptr) {
                return ge::GRAPH_FAILED;
            }
            workspaces[0] = static_cast<size_t>(workspace_bytes);
        }
    }

    if (tiling_key == INVALID_TILING_KEY) {
        return ge::GRAPH_FAILED;
    }
    context->SetTilingKey(tiling_key);
    context->SetBlockDim(block_dim);
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x_shape = context->GetInputShape(kInputX);
    const gert::Shape* scales_shape = context->GetInputShape(kInputScales);
    gert::Shape* y_shape = context->GetOutputShape(0);
    if (x_shape == nullptr || scales_shape == nullptr || y_shape == nullptr || x_shape->GetDimNum() != 2 ||
        scales_shape->GetDimNum() != 2) {
        return GRAPH_FAILED;
    }
    *y_shape = gert::Shape({x_shape->GetDim(0), scales_shape->GetDim(1)});
    return GRAPH_SUCCESS;
}

static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(kInputX));
    return ge::GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class KomodoCannW4A16Matmul : public OpDef {
public:
    explicit KomodoCannW4A16Matmul(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("packed_weight")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("scales")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("offsets")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("bias")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("group_size").Int();
        this->Attr("split_k").Int();
        this->Attr("base_m").Int();
        this->Attr("base_n").Int();
        this->Attr("base_k").Int();

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(KomodoCannW4A16Matmul);
}  // namespace ops
