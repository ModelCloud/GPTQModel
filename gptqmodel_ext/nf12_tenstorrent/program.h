// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#ifdef NF12_PJRT
#define NF12_CONFIG(index) nf12_config[index]
#define NF12_ACCESSORS(native_count) nf12_cb_count
#define NF12_ADDRESS(index) get_common_arg_val<uint32_t>(index)
#define NF12_PACKED_PAGE 4096
#define NF12_ESCAPE_PAGE 4096
#define NF12_RANGE(tasks, lo, hi)                                                   \
    const uint32_t core = get_absolute_logical_y() * nf12_grid_x + get_absolute_logical_x(); \
    const uint32_t start = (tasks) * core / nf12_core_count;                         \
    const uint32_t end = (tasks) * (core + 1) / nf12_core_count
#else
#define NF12_CONFIG(index) get_compile_time_arg_val(index)
#define NF12_ACCESSORS(native_count) native_count
#define NF12_ADDRESS(index) get_arg_val<uint32_t>(index)
#define NF12_PACKED_PAGE 1536
#define NF12_ESCAPE_PAGE 64
#define NF12_RANGE(tasks, lo, hi)              \
    const uint32_t start = get_arg_val<uint32_t>(lo); \
    const uint32_t end = get_arg_val<uint32_t>(hi)
#endif
