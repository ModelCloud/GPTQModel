#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import json
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
# Q4NX (cb = 3) decodes three trellis widths instead of one bitrate per row.
# ``step_bits`` is 4 * K + 2 * half_k, matching q4nx_supports_step_bits().
Q4NX_STEP_BITS = (10, 12, 14)
CC_ADA = 3
DEFAULT_Q4NX_SAMPLES = Path(__file__).resolve().parent / "exl3_kernel_map_q4nx_ada.json"
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


def q4nx_bits_index(step_bits: int) -> int:
    """Return the legacy ``bits`` row Q4NX falls back to: bits = K + half_k."""
    return step_bits // 4 + int(step_bits % 4 == 2)


def parse_q4nx_cell(text: str) -> tuple[int, int, int]:
    """Parse a ``"<rate>:<k>,<n>"`` fallback marker."""
    rate_text, _, key = text.partition(":")
    size_k, _, size_n = key.partition(",")
    return int(rate_text), int(size_k), int(size_n)


def load_q4nx_samples(
    path: Path,
) -> tuple[dict[int, dict[tuple[int, int], tuple[int, int]]], set[tuple[int, int, int]]]:
    """Read the measured Ada cells plus the explicitly declared fallbacks.

    A missing cell is only allowed when the sweep listed it under ``fallback``,
    so an incomplete measurement run cannot be mistaken for a tuned table.
    """

    payload = json.loads(path.read_text())
    capability = tuple(payload.get("capability", ()))
    device = str(payload.get("device", ""))
    if capability != (8, 9) or "4090" not in device:
        raise ValueError(
            f"{path} was measured on {device!r} {capability}, not an Ada RTX 4090 (sm_89); "
            "the Ada Q4NX grid must come from the 4090 sweep"
        )

    overrides: dict[int, dict[tuple[int, int], tuple[int, int]]] = {}
    for rate_text, cells in payload.get("rates", {}).items():
        rate = int(rate_text)
        if rate not in Q4NX_STEP_BITS:
            raise ValueError(f"unexpected Q4NX step_bits {rate}")
        for key, value in cells.items():
            size_k, size_n = (int(part) for part in key.split(","))
            if len(value) != 2:
                raise ValueError(f"Q4NX cell {rate}:{size_k},{size_n} must be [shape_idx, num_sms], got {value!r}")
            shape_idx, num_sms = (int(item) for item in value)
            if not 1 <= shape_idx <= 7:
                raise ValueError(f"Q4NX cell {rate}:{size_k},{size_n} has out-of-range shape {shape_idx}")
            if not 1 <= num_sms <= 255:
                raise ValueError(f"Q4NX cell {rate}:{size_k},{size_n} has out-of-range num_sms {num_sms}")
            overrides.setdefault(rate, {})[(size_k, size_n)] = (shape_idx, num_sms)

    fallbacks = {parse_q4nx_cell(item) for item in payload.get("fallback", [])}
    for rate, size_k, size_n in fallbacks:
        if rate not in Q4NX_STEP_BITS:
            raise ValueError(f"unexpected Q4NX fallback rate {rate}")
        if (size_k, size_n) in overrides.get(rate, {}):
            raise ValueError(f"Q4NX cell {rate}:{size_k},{size_n} is both measured and declared a fallback")
    return overrides, fallbacks


def q4nx_grid(blocks: dict[int, PackedBlock]) -> set[tuple[int, int, int]]:
    """Distinct (rate, k, n) cells; a cell feeds every block whose axes hold it."""

    return {
        (rate, size_k, size_n)
        for block in blocks.values()
        for rate in Q4NX_STEP_BITS
        for size_k in block.k_axis
        for size_n in block.n_axis
    }


def validate_q4nx_coverage(
    blocks: dict[int, PackedBlock],
    overrides: dict[int, dict[tuple[int, int], tuple[int, int]]],
    fallbacks: set[tuple[int, int, int]],
) -> None:
    grid = q4nx_grid(blocks)
    measured = {(rate, size_k, size_n) for rate, cells in overrides.items() for size_k, size_n in cells}
    unknown = (measured | fallbacks) - grid
    if unknown:
        raise ValueError(f"Q4NX cells are not on the packed grid: {sorted(unknown)}")
    missing = grid - measured - fallbacks
    if missing:
        raise ValueError(
            f"{len(missing)} Ada Q4NX cells are neither measured nor declared fallback: {sorted(missing)}"
        )


def packed_entry(block: PackedBlock, cc: int, bits: int, size_k: int, size_n: int) -> int:
    cc_idx = block.cc_values.index(cc)
    bit_idx = block.bit_values.index(bits)
    k_idx = block.k_axis.index(size_k)
    n_idx = block.n_axis.index(size_n)
    flat_idx = ((((cc_idx * len(block.bit_values)) + bit_idx) * len(block.k_axis)) + k_idx) * len(block.n_axis) + n_idx
    return block.payload[flat_idx]


def q4nx_ada_entry(
    block: PackedBlock,
    step_bits: int,
    size_k: int,
    size_n: int,
    ada_overrides: dict[int, dict[tuple[int, int], tuple[int, int]]],
    fallbacks: set[tuple[int, int, int]],
) -> int:
    measured = ada_overrides.get(step_bits, {}).get((size_k, size_n))
    if measured is not None:
        return pack_row(*measured)
    if (step_bits, size_k, size_n) not in fallbacks:
        raise ValueError(f"Ada Q4NX cell {step_bits}:{size_k},{size_n} is not measured or declared fallback")
    return packed_entry(block, CC_ADA, q4nx_bits_index(step_bits), size_k, size_n)


def build_q4nx_block(
    block: PackedBlock,
    ada_overrides: dict[int, dict[tuple[int, int], tuple[int, int]]],
    fallbacks: set[tuple[int, int, int]],
) -> tuple[int, ...]:
    """Pack [cc][q4nx rate][k][n]; only the Ada rows are re-measured."""

    payload: list[int] = []
    for cc in block.cc_values:
        for step_bits in Q4NX_STEP_BITS:
            bits = q4nx_bits_index(step_bits)
            for size_k in block.k_axis:
                for size_n in block.n_axis:
                    if cc == CC_ADA:
                        payload.append(q4nx_ada_entry(block, step_bits, size_k, size_n, ada_overrides, fallbacks))
                    else:
                        payload.append(packed_entry(block, cc, bits, size_k, size_n))
    return tuple(payload)


def validate_q4nx_block(
    block: PackedBlock,
    q4nx: tuple[int, ...],
    ada_overrides: dict[int, dict[tuple[int, int], tuple[int, int]]],
    fallbacks: set[tuple[int, int, int]],
) -> None:
    """Non-Ada rows must stay byte-identical to the three-instruction samples."""

    index = 0
    for cc in block.cc_values:
        for step_bits in Q4NX_STEP_BITS:
            bits = q4nx_bits_index(step_bits)
            for size_k in block.k_axis:
                for size_n in block.n_axis:
                    if cc == CC_ADA:
                        expected = q4nx_ada_entry(block, step_bits, size_k, size_n, ada_overrides, fallbacks)
                    else:
                        expected = packed_entry(block, cc, bits, size_k, size_n)
                    if q4nx[index] != expected:
                        raise ValueError(
                            f"q4nx row mismatch for samples_{block.mod} cc={cc} step_bits={step_bits} "
                            f"k={size_k} n={size_n}: {q4nx[index]:#06x} != {expected:#06x}"
                        )
                    index += 1


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


def format_fallback_comment(fallbacks: set[tuple[int, int, int]]) -> list[str]:
    """Render the declared fallbacks so the header itself states what was not measured."""

    if not fallbacks:
        return ["// Q4NX Ada coverage: every (rate, k, n) cell below is a measured sample."]
    tokens = [f"{rate}:{size_k},{size_n}" for rate, size_k, size_n in sorted(fallbacks)]
    lines = [
        f"// Q4NX Ada coverage: {len(fallbacks)} (rate, k, n) cells were not measured and keep the",
        "// pre-Q4NX bitrate-row sample:",
    ]
    current = "//   "
    for token in tokens:
        if len(current) + len(token) + 1 > 100:
            lines.append(current.rstrip())
            current = "//   "
        current += token + " "
    lines.append(current.rstrip())
    return lines


def render_header(
    blocks: dict[int, PackedBlock],
    q4nx_blocks: dict[int, tuple[int, ...]],
    q4nx_fallbacks: set[tuple[int, int, int]],
    q4nx_measured: int,
    q4nx_cells: int,
) -> str:
    block128 = blocks[128]
    block256 = blocks[256]
    block512 = blocks[512]

    sections = [
        "#pragma once",
        "",
        "#include <cstdint>",
        "",
        "// Generated by scripts/generate_exl3_kernel_map_packed.py.",
        "// Encodes the EXL3 tuning samples as dense [cc][bits][k][n] grids, plus a",
        "// [cc][q4nx rate][k][n] grid for the Q4NX codebook (cb = 3), whose three",
        "// trellis widths do not line up with the bitrate rows above.",
        "//",
        f"// Q4NX Ada cells measured on an RTX 4090 (sm_89): {q4nx_measured}/{q4nx_cells}.",
        *format_fallback_comment(q4nx_fallbacks),
        "",
        "namespace exl3_packed {",
        "",
        f"constexpr int cc_count = {len(block128.cc_values)};",
        f"constexpr int bit_count = {len(block128.bit_values)};",
        f"constexpr int k_axis_len = {len(block128.k_axis)};",
        f"constexpr int n_axis_len_128 = {len(block128.n_axis)};",
        f"constexpr int n_axis_len_256 = {len(block256.n_axis)};",
        f"constexpr int n_axis_len_512 = {len(block512.n_axis)};",
        f"constexpr int q4nx_rate_count = {len(Q4NX_STEP_BITS)};",
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
        format_int_array("q4nx_rate_step_bits", Q4NX_STEP_BITS),
        "",
        format_u16_array("samples_128", block128.payload),
        "",
        format_u16_array("samples_256", block256.payload),
        "",
        format_u16_array("samples_512", block512.payload),
        "",
        format_u16_array("samples_128_q4nx", q4nx_blocks[128]),
        "",
        format_u16_array("samples_256_q4nx", q4nx_blocks[256]),
        "",
        format_u16_array("samples_512_q4nx", q4nx_blocks[512]),
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
        "--q4nx-samples",
        type=Path,
        default=DEFAULT_Q4NX_SAMPLES,
        help="Ada Q4NX measurements; a missing cell must be declared fallback and then inherits the legacy row.",
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
    q4nx_overrides, q4nx_fallbacks = load_q4nx_samples(args.q4nx_samples)
    validate_q4nx_coverage(packed, q4nx_overrides, q4nx_fallbacks)
    q4nx_blocks = {
        mod: build_q4nx_block(block, q4nx_overrides, q4nx_fallbacks) for mod, block in packed.items()
    }
    for mod, block in packed.items():
        validate_q4nx_block(block, q4nx_blocks[mod], q4nx_overrides, q4nx_fallbacks)
    q4nx_cells = len(q4nx_grid(packed))
    q4nx_measured = q4nx_cells - len(q4nx_fallbacks)
    args.output.write_text(
        render_header(packed, q4nx_blocks, q4nx_fallbacks, q4nx_measured, q4nx_cells)
    )

    payload_entries = sum(len(block.payload) for block in packed.values())
    print(
        f"validated legacy lookup from {source} and wrote {args.output} "
        f"({payload_entries} packed entries across {len(packed)} blocks)"
    )
    print(f"Q4NX Ada cells: {q4nx_measured} measured, {len(q4nx_fallbacks)} declared fallback, {q4nx_cells} total")


if __name__ == "__main__":
    main()
