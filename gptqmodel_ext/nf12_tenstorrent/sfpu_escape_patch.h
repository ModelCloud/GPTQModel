// SPDX-FileCopyrightText: 2026 ModelCloud.ai
// SPDX-License-Identifier: Apache-2.0

#pragma once

template <typename Header, typename Escape>
struct EscapePatches {
    const Header& headers;
    const Escape& escapes;
    uint32_t scratch;
#ifdef NF12_PJRT
    uint32_t first_escape_page;
#endif
    uint32_t loaded_header_page = uint32_t(-1);
    uint32_t loaded_escape_page = uint32_t(-1);

    template <typename Accessor>
    static void read(const Accessor& stream, uint32_t page, uint32_t destination) {
#ifdef NF12_PJRT
        noc_async_read(stream.get_noc_addr(page / 64, page % 64 * 64), destination, 64);
#else
        noc_async_read_page(page, stream, destination);
#endif
        noc_async_read_barrier();
    }

    void apply(uint32_t block, uint32_t address) {
        constexpr uint32_t word_count = 16;
        constexpr uint32_t headers_per_page = word_count / 2;
        const auto* header_words = reinterpret_cast<volatile const uint32_t*>(scratch);
        const uint32_t escape_address = scratch + 64;
        const auto* escape_words = reinterpret_cast<volatile const uint32_t*>(escape_address);
        auto* decoded = reinterpret_cast<volatile uint16_t*>(address);

        const uint32_t header_page = block / headers_per_page;
        if (header_page != loaded_header_page) {
            read(headers, header_page, scratch);
            loaded_header_page = header_page;
        }
        const uint32_t index = header_words[2 * (block % headers_per_page)];
        const uint32_t count = header_words[2 * (block % headers_per_page) + 1];
        for (uint32_t i = index; i < index + count; ++i) {
            const uint32_t page = i / word_count;
            if (page != loaded_escape_page) {
#ifdef NF12_PJRT
                read(escapes, first_escape_page + page, escape_address);
#else
                read(escapes, page, escape_address);
#endif
                loaded_escape_page = page;
            }
            const uint32_t patch = escape_words[i % word_count];
            decoded[patch & 0xfffu] = patch >> 12;
        }
    }
};
