// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#pragma once

template <typename Header, typename Escape>
struct EscapePatches {
    const Header& headers;
    const Escape& escapes;
    uint32_t scratch;
    uint32_t loaded_header_page = uint32_t(-1);
    uint32_t loaded_escape_page = uint32_t(-1);

    void apply(uint32_t block, uint32_t address) {
        constexpr uint32_t word_count = 16;
        constexpr uint32_t headers_per_page = word_count / 2;
        const auto* header_words = reinterpret_cast<volatile const uint32_t*>(scratch);
        const uint32_t escape_address = scratch + 64;
        const auto* escape_words = reinterpret_cast<volatile const uint32_t*>(escape_address);
        auto* decoded = reinterpret_cast<volatile uint16_t*>(address);

        const uint32_t header_page = block / headers_per_page;
        if (header_page != loaded_header_page) {
            noc_async_read_page(header_page, headers, scratch);
            noc_async_read_barrier();
            loaded_header_page = header_page;
        }
        const uint32_t index = header_words[2 * (block % headers_per_page)];
        const uint32_t count = header_words[2 * (block % headers_per_page) + 1];
        for (uint32_t i = index; i < index + count; ++i) {
            const uint32_t page = i / word_count;
            if (page != loaded_escape_page) {
                noc_async_read_page(page, escapes, escape_address);
                noc_async_read_barrier();
                loaded_escape_page = page;
            }
            const uint32_t patch = escape_words[i % word_count];
            decoded[patch & 0xfffu] = patch >> 12;
        }
    }
};
