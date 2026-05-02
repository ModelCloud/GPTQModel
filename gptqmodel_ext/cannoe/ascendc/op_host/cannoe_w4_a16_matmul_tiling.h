#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(CannoeW4A16MatmulTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, rows);
  TILING_DATA_FIELD_DEF(uint32_t, in_features);
  TILING_DATA_FIELD_DEF(uint32_t, out_features);
  TILING_DATA_FIELD_DEF(uint32_t, group_size);
  TILING_DATA_FIELD_DEF(uint32_t, has_bias);
  TILING_DATA_FIELD_DEF(uint32_t, zero_offsets);
  TILING_DATA_FIELD_DEF(uint32_t, kernel_mode);
  TILING_DATA_FIELD_DEF(uint32_t, split_k);
  TILING_DATA_FIELD_DEF(uint32_t, base_m);
  TILING_DATA_FIELD_DEF(uint32_t, base_n);
  TILING_DATA_FIELD_DEF(uint32_t, base_k);
  TILING_DATA_FIELD_DEF(uint32_t, total_outputs);
  TILING_DATA_FIELD_DEF(uint32_t, block_dim);
  TILING_DATA_FIELD_DEF(uint32_t, staging_blocks);
  TILING_DATA_FIELD_DEF(uint32_t, staging_slots);
  TILING_DATA_FIELD_DEF(uint32_t, staging_tile_bytes);
  TILING_DATA_FIELD_DEF(uint32_t, staging_workspace_bytes);
  TILING_DATA_FIELD_DEF(uint32_t, staging_workspace_offset);
  TILING_DATA_FIELD_DEF(uint32_t, cube_workspace_bytes);
  TILING_DATA_FIELD_DEF(uint32_t, ub_bytes);
  TILING_DATA_FIELD_DEF(uint32_t, l1_bytes);
  TILING_DATA_FIELD_DEF(uint32_t, l0a_bytes);
  TILING_DATA_FIELD_DEF(uint32_t, l0b_bytes);
  TILING_DATA_FIELD_DEF(uint32_t, l0c_bytes);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(CannoeW4A16Matmul, CannoeW4A16MatmulTilingData)
}  // namespace optiling
