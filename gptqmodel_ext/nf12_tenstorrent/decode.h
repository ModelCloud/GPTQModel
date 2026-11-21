// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"

struct Group {
    uint32_t meta;
    uint32_t low0;
    uint32_t low1;
};

inline Group load_group(const volatile uint32_t* atom, uint32_t face, uint32_t lane) {
    const uint32_t low = 128 * (1 + face / 2) + 4 * lane + 2 * (face % 2);
    return {atom[4 * lane + face], atom[low], atom[low + 1]};
}

template <typename Accessor>
inline void fetch_escape(const Group& group, const Accessor& escapes, uint32_t dst) {
    if (group.meta == 0) {
        // Escape pages contain four canonical 16-byte entries. A full page
        // keeps the DRAM/L1 transfer aligned on Blackhole.
        noc_async_read_page(group.low0 / 4, escapes, dst);
    }
}

inline uint16_t decode_slot(const Group& group, uint32_t slot, uint32_t base, uint32_t escape_addr) {
    if (group.meta == 0) {
        const auto* bits = reinterpret_cast<volatile const uint16_t*>(escape_addr);
        return bits[8 * (group.low0 % 4) + slot];
    }
    const uint32_t pair = group.meta >> (8 * (slot / 2));
    const uint32_t high = slot % 2 ? (((pair << 1) & 0x80) | ((pair >> 3) & 7)) : (pair & 0x87);
    const uint32_t low = slot < 4 ? group.low0 : group.low1;
    return ((high + base) << 8) | ((low >> (8 * (slot % 4))) & 255);
}

inline uint32_t tile_slot(uint32_t row, uint32_t col) {
    return (row / 16 * 2 + col / 16) * 256 + (row % 16) * 16 + col % 16;
}
