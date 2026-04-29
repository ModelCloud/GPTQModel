#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import pcre

BLOCK_RE = pcre.compile(
    r"struct TSample samples_(\d+)\[\]\s*=\s*\{(.*?)\n\};",
    flags=pcre.Flag.DOTALL,
)
ROW_RE = pcre.compile(
    r"\{\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\s*\}"
)
EXLLAMAV3_ORIGINAL_COMMIT = "ba1ad9ac66670785c0ca95b0f1ab3ad044fda7c6"
EXLLAMAV3_ORIGINAL_LEGACY_HEADER_URL = (
    "https://raw.githubusercontent.com/turboderp-org/exllamav3/"
    f"{EXLLAMAV3_ORIGINAL_COMMIT}/exllamav3/exllamav3_ext/quant/exl3_kernel_map_samples.cuh"
)


@dataclass(frozen=True)
class LegacyRow:
    cc: int
    bits: int
    m: int
    k: int
    n: int
    shape_idx: int
    num_sms: int


@dataclass(frozen=True)
class PackedBlock:
    mod: int
    cc_values: tuple[int, ...]
    bit_values: tuple[int, ...]
    k_axis: tuple[int, ...]
    n_axis: tuple[int, ...]
    payload: tuple[int, ...]
    row_lookup: dict[tuple[int, int, int, int], LegacyRow]


def parse_legacy_text(text: str) -> dict[int, list[LegacyRow]]:
    blocks: dict[int, list[LegacyRow]] = {}

    for mod_text, body in BLOCK_RE.findall(text):
        mod = int(mod_text)
        rows: list[LegacyRow] = []
        for match in ROW_RE.findall(body):
            row = LegacyRow(*(int(value) for value in match))
            if row.bits == 0:
                continue
            rows.append(row)
        blocks[mod] = rows

    expected_blocks = {128, 256, 512}
    if set(blocks) != expected_blocks:
        raise ValueError(f"expected blocks {sorted(expected_blocks)}, found {sorted(blocks)}")

    return blocks


def parse_legacy_header(path: Path) -> dict[int, list[LegacyRow]]:
    return parse_legacy_text(path.read_text())


def download_legacy_header(url: str = EXLLAMAV3_ORIGINAL_LEGACY_HEADER_URL, timeout: int = 30) -> str:
    with urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8")


def pack_row(shape_idx: int, num_sms: int) -> int:
    if not (0 < shape_idx < 256):
        raise ValueError(f"shape_idx out of range: {shape_idx}")
    if not (0 < num_sms < 256):
        raise ValueError(f"num_sms out of range: {num_sms}")
    return (shape_idx << 8) | num_sms


def build_packed_block(mod: int, rows: list[LegacyRow]) -> PackedBlock:
    cc_values = tuple(sorted({row.cc for row in rows}))
    bit_values = tuple(sorted({row.bits for row in rows}))
    k_axis = tuple(sorted({row.k for row in rows}))
    n_axis = tuple(sorted({row.n for row in rows}))

    if cc_values != (2, 3, 4):
        raise ValueError(f"unexpected cc axis for {mod}: {cc_values}")
    if bit_values != (1, 2, 3, 4, 5, 6, 7, 8):
        raise ValueError(f"unexpected bit axis for {mod}: {bit_values}")
    if any(row.m != 1 for row in rows):
        raise ValueError(f"unexpected m values in samples_{mod}")

    expected = len(cc_values) * len(bit_values) * len(k_axis) * len(n_axis)
    if len(rows) != expected:
        raise ValueError(f"samples_{mod} is not a full grid: got {len(rows)}, expected {expected}")

    row_lookup: dict[tuple[int, int, int, int], LegacyRow] = {}
    for row in rows:
        key = (row.cc, row.bits, row.k, row.n)
        if key in row_lookup:
            raise ValueError(f"duplicate row for {key} in samples_{mod}")
        row_lookup[key] = row

    payload: list[int] = []
    for cc in cc_values:
        for bits in bit_values:
            for k in k_axis:
                for n in n_axis:
                    key = (cc, bits, k, n)
                    row = row_lookup.get(key)
                    if row is None:
                        raise ValueError(f"missing row for {key} in samples_{mod}")
                    payload.append(pack_row(row.shape_idx, row.num_sms))

    return PackedBlock(
        mod=mod,
        cc_values=cc_values,
        bit_values=bit_values,
        k_axis=k_axis,
        n_axis=n_axis,
        payload=tuple(payload),
        row_lookup=row_lookup,
    )


def nearest_axis_index(axis: tuple[int, ...], value: int) -> int:
    best_idx = 0
    best_dist = abs(value - axis[0])
    for idx in range(1, len(axis)):
        dist = abs(value - axis[idx])
        if dist < best_dist:
            best_dist = dist
            best_idx = idx
    return best_idx


def packed_lookup(block: PackedBlock, cc: int, bits: int, size_k: int, size_n: int) -> tuple[int, int]:
    try:
        cc_idx = block.cc_values.index(cc)
        bit_idx = block.bit_values.index(bits)
    except ValueError as exc:
        raise ValueError(f"unsupported lookup key {(cc, bits)} for block {block.mod}") from exc

    k_idx = nearest_axis_index(block.k_axis, size_k)
    n_idx = nearest_axis_index(block.n_axis, size_n)
    flat_idx = ((((cc_idx * len(block.bit_values)) + bit_idx) * len(block.k_axis)) + k_idx) * len(block.n_axis) + n_idx
    packed = block.payload[flat_idx]
    return packed >> 8, packed & 0xFF


def legacy_lookup(rows: list[LegacyRow], cc: int, bits: int, size_k: int, size_n: int) -> tuple[int, int]:
    best_row: LegacyRow | None = None
    best_dist: int | None = None
    for row in rows:
        if row.cc != cc or row.bits != bits:
            continue
        distk = size_k - row.k
        distn = size_n - row.n
        dist = distk * distk + distn * distn
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_row = row

    if best_row is None:
        raise ValueError(f"no legacy row for {(cc, bits, size_k, size_n)}")
    return best_row.shape_idx, best_row.num_sms


def transition_points(axis: tuple[int, ...]) -> list[int]:
    points = {1, axis[-1] + 1}
    for left, right in zip(axis, axis[1:]):
        midpoint = (left + right) // 2
        points.add(midpoint)
        points.add(midpoint + 1)
    return sorted(points)


def block_domain_values(mod: int, max_value: int) -> list[int]:
    values = []
    value = mod
    while value <= max_value:
        if mod == 512:
            valid = value % 512 == 0
        elif mod == 256:
            valid = value % 256 == 0 and value % 512 != 0
        elif mod == 128:
            valid = value % 128 == 0 and value % 256 != 0
        else:
            raise ValueError(f"unexpected block modulus: {mod}")
        if valid:
            values.append(value)
        value += 128
    return values


def domain_transition_points(mod: int, axis: tuple[int, ...]) -> list[int]:
    domain = block_domain_values(mod, axis[-1] + mod)
    points = {domain[0], domain[-1]}

    for left, right in zip(axis, axis[1:]):
        midpoint = (left + right) // 2
        lower_idx = bisect.bisect_right(domain, midpoint) - 1
        upper_idx = bisect.bisect_right(domain, midpoint)
        if lower_idx >= 0:
            points.add(domain[lower_idx])
        if upper_idx < len(domain):
            points.add(domain[upper_idx])

    return sorted(points)


def validate_row_exactness(legacy: dict[int, list[LegacyRow]], packed: dict[int, PackedBlock]) -> None:
    for mod, block in packed.items():
        for row in legacy[mod]:
            expected = pack_row(row.shape_idx, row.num_sms)
            cc_idx = block.cc_values.index(row.cc)
            bit_idx = block.bit_values.index(row.bits)
            k_idx = block.k_axis.index(row.k)
            n_idx = block.n_axis.index(row.n)
            flat_idx = ((((cc_idx * len(block.bit_values)) + bit_idx) * len(block.k_axis)) + k_idx) * len(block.n_axis) + n_idx
            actual = block.payload[flat_idx]
            if actual != expected:
                raise ValueError(f"packed row mismatch for samples_{mod}: {row} -> {actual:#06x} != {expected:#06x}")


def validate_lookup_equivalence(legacy: dict[int, list[LegacyRow]], packed: dict[int, PackedBlock]) -> None:
    k_points = transition_points(packed[128].k_axis)

    for mod, block in packed.items():
        n_points = domain_transition_points(mod, block.n_axis)
        grouped_rows: dict[tuple[int, int], list[LegacyRow]] = {}
        for row in legacy[mod]:
            grouped_rows.setdefault((row.cc, row.bits), []).append(row)
        for cc in block.cc_values:
            for bits in block.bit_values:
                legacy_rows = grouped_rows[(cc, bits)]
                for size_k in k_points:
                    for size_n in n_points:
                        legacy_result = legacy_lookup(legacy_rows, cc, bits, size_k, size_n)
                        packed_result = packed_lookup(block, cc, bits, size_k, size_n)
                        if packed_result != legacy_result:
                            raise ValueError(
                                "lookup mismatch for "
                                f"samples_{mod} cc={cc} bits={bits} size_k={size_k} size_n={size_n}: "
                                f"{packed_result} != {legacy_result}"
                            )


def format_int_array(name: str, values: tuple[int, ...], values_per_line: int = 8) -> str:
    lines = [f"constexpr int {name}[] = {{"]
    for idx in range(0, len(values), values_per_line):
        chunk = ", ".join(str(value) for value in values[idx:idx + values_per_line])
        lines.append(f"    {chunk},")
    lines.append("};")
    return "\n".join(lines)


def format_u16_array(name: str, values: tuple[int, ...], values_per_line: int = 24) -> str:
    lines = [f"constexpr uint16_t {name}[] = {{"]
    for idx in range(0, len(values), values_per_line):
        chunk = ", ".join(f"0x{value:04x}" for value in values[idx:idx + values_per_line])
        lines.append(f"    {chunk},")
    lines.append("};")
    return "\n".join(lines)


def render_header(blocks: dict[int, PackedBlock]) -> str:
    block128 = blocks[128]
    block256 = blocks[256]
    block512 = blocks[512]

    sections = [
        "#pragma once",
        "",
        "#include <cstdint>",
        "",
        "// Generated by scripts/generate_exl3_kernel_map_packed.py.",
        "// Encodes the EXL3 tuning samples as dense [cc][bits][k][n] grids.",
        "",
        "namespace exl3_packed {",
        "",
        f"constexpr int cc_count = {len(block128.cc_values)};",
        f"constexpr int bit_count = {len(block128.bit_values)};",
        f"constexpr int k_axis_len = {len(block128.k_axis)};",
        f"constexpr int n_axis_len_128 = {len(block128.n_axis)};",
        f"constexpr int n_axis_len_256 = {len(block256.n_axis)};",
        f"constexpr int n_axis_len_512 = {len(block512.n_axis)};",
        "",
        format_int_array("cc_values", block128.cc_values),
        "",
        format_int_array("bit_values", block128.bit_values),
        "",
        format_int_array("k_axis", block128.k_axis),
        "",
        format_int_array("n_axis_128", block128.n_axis),
        "",
        format_int_array("n_axis_256", block256.n_axis),
        "",
        format_int_array("n_axis_512", block512.n_axis),
        "",
        format_u16_array("samples_128", block128.payload),
        "",
        format_u16_array("samples_256", block256.payload),
        "",
        format_u16_array("samples_512", block512.payload),
        "",
        "}  // namespace exl3_packed",
        "",
    ]
    return "\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the packed EXL3 kernel-map header from the legacy row table.")
    parser.add_argument(
        "--legacy",
        type=Path,
        help="Optional local path to the legacy row-wise samples header.",
    )
    parser.add_argument(
        "--url",
        default=EXLLAMAV3_ORIGINAL_LEGACY_HEADER_URL,
        help="Raw URL for the original ExLlamaV3 legacy row-wise samples header.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("gptqmodel_ext/exllamav3/quant/exl3_kernel_map_packed.cuh"),
        help="Path to the generated packed-grid header.",
    )
    args = parser.parse_args()

    if args.legacy is not None:
        legacy = parse_legacy_header(args.legacy)
        source = str(args.legacy)
    else:
        legacy = parse_legacy_text(download_legacy_header(args.url))
        source = args.url

    packed = {mod: build_packed_block(mod, rows) for mod, rows in sorted(legacy.items())}
    validate_row_exactness(legacy, packed)
    validate_lookup_equivalence(legacy, packed)
    args.output.write_text(render_header(packed))

    payload_entries = sum(len(block.payload) for block in packed.values())
    print(
        f"validated legacy lookup from {source} and wrote {args.output} "
        f"({payload_entries} packed entries across {len(packed)} blocks)"
    )


if __name__ == "__main__":
    main()
