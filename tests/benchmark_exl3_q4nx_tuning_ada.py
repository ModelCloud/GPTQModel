#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Measure and re-tune the EXL3 Q4NX kernel table on Ada (sm_89).

The packed table in ``exl3_kernel_map_packed.cuh`` stores one
``(shape_idx, num_sms)`` sample per ``[cc][bits][k][n]`` cell.  It was measured
for the three-instruction codebooks, so Q4NX currently inherits those samples
even though it decodes a different trellis width and runs different kernels.

This script is the measurement half of the fix:

* ``sweep``   - for every ``(rate, k, n, m)`` cell of the table axes, time every
                compatible kernel shape through the pinned-config hook and emit
                the ``argmin`` table (plus a ``num_sms`` sweep for the winner).
* ``win``     - auto-selected dispatch with the new table versus the dispatch
                the old table produced, both measured through the public op.
* ``verify``  - Q4NX numerics against the reconstruct + HGEMM reference.

Timings use CUDA events with 5 warmups and the median of 20 launches.

Typical use on a 4090::

    export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4
    export PYTHONPATH=$PWD HF_HUB_OFFLINE=1
    export GPTQMODEL_EXLLAMAV3_BUILD_ROOT=$PWD/.extbuild
    python tests/benchmark_exl3_q4nx_tuning_ada.py sweep \
        --max-weight-gb 6 --resume --emit-json /tmp/q4nx_ada.json
    python tests/benchmark_exl3_q4nx_tuning_ada.py win --old-header /tmp/base_packed.cuh

``CUDA_DEVICE_ORDER=PCI_BUS_ID`` is required: with the default CUDA ordering the
container's GPU index 4 is the RTX 5090, and the sweep would refuse to run.
Every measured cell is written after it completes, so an interrupted sweep can
be resumed with ``--resume`` instead of restarting from zero.

``sweep`` needs the ``exl3_gemm_force_tuned_config`` entry point that
``exl3_kernel_map.cu`` exports for exactly this purpose; without it the script
refuses to run instead of silently measuring the tuned dispatch.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import re
import statistics
import sys
import time
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKED_HEADER = REPO_ROOT / "gptqmodel_ext/exllamav3/quant/exl3_kernel_map_packed.cuh"

# Mirrors EXL3_GEMM_TILESIZE_M / _K / _N in exl3_kernel_map.cuh.
SHAPE_TILES = {
    1: (16, 16, 128),
    2: (16, 32, 128),
    3: (16, 32, 256),
    4: (16, 16, 512),
    5: (32, 32, 128),
    6: (48, 32, 128),
    7: (64, 32, 128),
}

# Q4NX rates keyed by the packed trellis width, matching q4nx_supports_step_bits().
Q4NX_STEP_BITS = {10: 2.5, 12: 3.0, 14: 3.5}

WIN_SHAPES = ((2048, 2048), (4096, 4096), (4096, 8192))
WIN_ROWS = (16, 64, 128, 512, 2048)


def _log(message: str) -> None:
    print(message, flush=True)


def _require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if torch.cuda.device_count() != 1:
        raise SystemExit(
            "pin the sweep to one device with CUDA_VISIBLE_DEVICES; "
            f"{torch.cuda.device_count()} devices are visible"
        )
    return torch.device("cuda", torch.cuda.current_device())


def _prewarm() -> None:
    from gptqmodel.exllamav3.ext import prewarm_exllamav3_extension

    if not prewarm_exllamav3_extension():
        raise SystemExit("the EXL3 torch.ops extension could not be built")


def _sweep_library(explicit: str | None) -> ctypes.CDLL:
    """Load the extension binary that ``prewarm`` just built as a plain .so.

    Only the current fingerprint's directory may be loaded: a stale build of the
    same torch.ops namespace aborts the process on dlopen.
    """
    if explicit:
        candidates = [Path(explicit)]
    else:
        roots = []
        from gptqmodel.exllamav3.ext import _EXLLAMAV3_TORCH_OPS_EXTENSION

        roots.append(_EXLLAMAV3_TORCH_OPS_EXTENSION.build_root())
        candidates = []
        for root in roots:
            if root.is_dir():
                candidates.extend(sorted(root.glob("gptqmodel_exllamav3*.so"), key=lambda path: path.stat().st_mtime))
    for path in reversed(candidates):
        if not path.is_file():
            continue
        try:
            library = ctypes.CDLL(str(path))
        except OSError:
            continue
        if hasattr(library, "exl3_gemm_force_tuned_config"):
            return library
    raise SystemExit(
        "could not find `exl3_gemm_force_tuned_config`; build the extension and pass --sweep-lib"
    )


class PinnedConfig:
    """Pin (shape_idx, num_sms) for the tuned dispatch through the C hook."""

    def __init__(self, library: ctypes.CDLL) -> None:
        self._library = library
        self._library.exl3_gemm_force_tuned_config.argtypes = [ctypes.c_int, ctypes.c_int]
        self._library.exl3_gemm_force_tuned_config.restype = None

    def set(self, shape_idx: int, num_sms: int) -> None:
        self._library.exl3_gemm_force_tuned_config(int(shape_idx), int(num_sms))

    def clear(self) -> None:
        self.set(-1, 0)


def _packed_projection(k: int, n: int, step_bits: int, seed: int = 37) -> tuple[torch.Tensor, ...]:
    """Random but legal Q4NX projection: any 16-bit trellis word decodes to a state."""
    generator = torch.Generator().manual_seed(seed)
    # The packed trellis is 16 * K + 8 * half_k words wide, i.e. 4 * step_bits.
    trellis = torch.randint(
        -32768, 32768, (k // 16, n // 16, 4 * step_bits), generator=generator, dtype=torch.int16
    ).cuda()
    su = ((2 * torch.randint(0, 2, (k,), generator=generator) - 1) * 0.5).half().cuda()
    sv = (torch.randn(n, generator=generator) * 0.03).half().cuda()
    return trellis, su, sv


def _input(rows: int, k: int, seed: int = 19) -> torch.Tensor:
    return (torch.randn((rows, k), generator=torch.Generator().manual_seed(seed)) * 0.1).half().cuda()


def _time_ms(run, warmup: int, reps: int) -> float:
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def _compatible(shape: int, k: int, n: int) -> bool:
    _, tilesize_k, tilesize_n = SHAPE_TILES[shape]
    return k % tilesize_k == 0 and n % tilesize_n == 0


def _pick_shape(per_row: dict[int, dict[int, float]], tolerance: float) -> int:
    """Shape with the smallest worst-case slowdown across the measured rows.

    The packed table is only read while ``size_m <= 16``, so the sweep measures
    the endpoints of that band.  Shapes 5-7 tile 32, 48 or 64 rows and pad a
    16-row call, which shows up as a large m=16 penalty; scoring each shape by
    its worst ratio to the row's own best keeps a padding shape from winning on
    one row at the cost of the other.  Among shapes within ``tolerance`` of the
    best worst-case score, the one that pads the fewest rows wins, and the
    better score breaks a tie between shapes with the same row tile.  The
    tolerance is documented with the emitted table.
    """

    best = {rows: min(timings.values()) for rows, timings in per_row.items() if timings}
    score: dict[int, float] = {}
    for rows, timings in per_row.items():
        for shape, elapsed in timings.items():
            score[shape] = max(score.get(shape, 0.0), elapsed / best[rows])
    floor = min(score.values())
    candidates = [shape for shape, value in score.items() if value <= floor * (1.0 + tolerance)]
    return min(candidates, key=lambda shape: (SHAPE_TILES[shape][0], score[shape], shape))


class HeaderTable:
    """Read the packed ``[cc][bits][k][n]`` grid back out of a header."""

    _BLOCK = re.compile(r"constexpr uint16_t samples_(\d+)\[\] = \{(.*?)\n\};", re.S)
    _INTS = re.compile(r"constexpr int (k_axis|n_axis_128|n_axis_256|n_axis_512)\[\] = \{(.*?)\n\};", re.S)

    def __init__(self, text: str) -> None:
        self.axes = {}
        for name, body in self._INTS.findall(text):
            self.axes[name] = [int(value) for value in body.replace("\n", " ").split(",") if value.strip()]
        self.blocks = {}
        for mod, body in self._BLOCK.findall(text):
            payload = [int(value, 16) for value in body.replace("\n", " ").split(",") if value.strip()]
            self.blocks[int(mod)] = payload
        if set(self.blocks) != {128, 256, 512}:
            raise SystemExit("header does not contain the expected 128/256/512 blocks")
        self.k_axis = self.axes["k_axis"]

    @classmethod
    def from_path(cls, path: Path) -> "HeaderTable":
        return cls(path.read_text())

    def _table(self, size_n: int) -> tuple[str, int]:
        if size_n % 512 == 0:
            return "n_axis_512", 512
        if size_n % 256 == 0:
            return "n_axis_256", 256
        return "n_axis_128", 128

    def lookup(self, cc_idx: int, bits: int, size_k: int, size_n: int) -> tuple[int, int]:
        axis_name, mod = self._table(size_n)
        n_axis = self.axes[axis_name]
        k_idx = _nearest_index(self.k_axis, size_k)
        n_idx = _nearest_index(n_axis, size_n)
        flat = (((cc_idx * 8) + (bits - 1)) * len(self.k_axis) + k_idx) * len(n_axis) + n_idx
        packed = self.blocks[mod][flat]
        return packed >> 8, packed & 0xFF


def _nearest_index(axis: list[int], value: int) -> int:
    best_idx = 0
    best_dist = abs(value - axis[0])
    for idx in range(1, len(axis)):
        dist = abs(value - axis[idx])
        if dist < best_dist:
            best_dist = dist
            best_idx = idx
    return best_idx


def _old_choice(table: HeaderTable, cc_idx: int, step_bits: int, k: int, n: int) -> tuple[int, int]:
    """What the pre-Q4NX table produced: the lookup keys on ``K + half_k``."""
    # K = step_bits // 4 and half_k = step_bits % 4 == 2, so the pre-Q4NX table
    # was indexed with bits = K + half_k.
    bits = step_bits // 4 + (1 if step_bits % 4 == 2 else 0)
    return table.lookup(cc_idx, bits, k, n)


def _effective_num_sms(shape: int, k: int, n: int, cap: int) -> int:
    _, tilesize_k, tilesize_n = SHAPE_TILES[shape]
    max_slices = (k // tilesize_k) * (n // tilesize_n)
    return max(min(max_slices, cap), 1)


def _row_shape(rows: int) -> int:
    """Mirror the row-tile override in select_exl3_gemm_kernel_tuned()."""
    return 7 if rows > 48 else 6 if rows > 32 else 5 if rows > 16 else 0


def _call_gemm(op, weights, rows: int, k: int, n: int, step_bits: int, x, y, fp16_gemv=False):
    trellis, su, sv = weights
    return op(trellis, su, sv, Q4NX_STEP_BITS[step_bits], None, False, False, x, y, fp16_gemv, 0, True, q4nx=True)


def _sweep_cell(
    op, pinned: PinnedConfig, k: int, n: int, rows: int, step_bits: int, shapes, warmup, reps
) -> dict:
    weights = _packed_projection(k, n, step_bits)
    x = _input(rows, k)
    y = torch.empty((rows, n), device=x.device, dtype=torch.float32)
    timings = {}
    for shape in shapes:
        if not _compatible(shape, k, n):
            continue
        # The shape's own slice count is the natural cap; pinning a larger cap
        # would just be clamped by max_slices in the lookup.
        num_sms = _effective_num_sms(shape, k, n, 0xFFFF)
        pinned.set(shape, min(num_sms, 0xFF))
        try:
            _call_gemm(op, weights, rows, k, n, step_bits, x, y)
            torch.cuda.synchronize()
        except Exception as exc:  # noqa: BLE001 - a shape can be rejected by the kernel
            _log(f"    shape {shape} rejected for k={k} n={n} m={rows}: {exc}")
            continue
        timings[shape] = _time_ms(
            lambda: _call_gemm(op, weights, rows, k, n, step_bits, x, y), warmup, reps
        )
    pinned.clear()
    return timings


def _sweep_sms(
    op, pinned: PinnedConfig, k: int, n: int, rows: int, step_bits: int, shape: int, caps, warmup, reps
) -> dict:
    weights = _packed_projection(k, n, step_bits)
    x = _input(rows, k)
    y = torch.empty((rows, n), device=x.device, dtype=torch.float32)
    timings = {}
    for cap in caps:
        pinned.set(shape, cap)
        try:
            _call_gemm(op, weights, rows, k, n, step_bits, x, y)
            torch.cuda.synchronize()
        except Exception:  # noqa: BLE001
            continue
        timings[cap] = _time_ms(
            lambda: _call_gemm(op, weights, rows, k, n, step_bits, x, y), warmup, reps
        )
    pinned.clear()
    return timings


def _emit_payload(args, measured, skipped, rows_axis, k_axis, n_axis, rates, sms_caps, started) -> dict:
    """Serialise the sweep; every grid cell is either measured or an explicit fallback."""

    fallback = [
        f"{rate}:{k},{n}"
        for rate in rates
        for k in k_axis
        for n in n_axis
        if f"{k},{n}" not in measured[str(rate)]
    ]
    return {
        "device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "capability": list(torch.cuda.get_device_capability(torch.cuda.current_device())),
        "num_sms": torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count,
        "torch": torch.__version__,
        "warmup": args.warmup,
        "reps": args.reps,
        "tie_tolerance": args.tie_tolerance,
        "m_axis": rows_axis,
        "sms_caps": sms_caps,
        "axes": {"rates": rates, "k": k_axis, "n": n_axis},
        "grid_cells": len(rates) * len(k_axis) * len(n_axis),
        "measured_cells": sum(len(cells) for cells in measured.values()),
        "fallback": fallback,
        "skipped": skipped,
        "note": args.note,
        "started_utc": started,
        "rates": measured,
    }


def cmd_sweep(args) -> int:
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    started_monotonic = time.monotonic()
    device = _require_cuda()
    capability = torch.cuda.get_device_capability(device)
    if capability != (8, 9) and not args.allow_other_cc:
        raise SystemExit(f"sweep measures the Ada table; device is sm_{capability[0]}{capability[1]}")
    _prewarm()
    op = torch.ops.gptqmodel_exllamav3.bc_linear_exl3_run_configured
    pinned = PinnedConfig(_sweep_library(args.sweep_lib))

    header = HeaderTable.from_path(Path(args.header))
    n_axis = sorted({value for name in ("n_axis_128", "n_axis_256", "n_axis_512") for value in header.axes[name]})
    k_axis = header.k_axis
    if args.k_axis:
        k_axis = [value for value in k_axis if value in {int(item) for item in args.k_axis.split(",")}]
    if args.n_axis:
        n_axis = [value for value in n_axis if value in {int(item) for item in args.n_axis.split(",")}]
    rates = [int(rate) for rate in args.rates.split(",")]
    rows_axis = [int(value) for value in args.m_axis.split(",")]
    shapes = [int(value) for value in args.shapes.split(",")]
    sms_caps = [int(value) for value in args.sms_sweep.split(",")] if args.sms_sweep else []
    sms_rows = [int(value) for value in args.sms_m_axis.split(",")] if args.sms_sweep else []

    _log(
        f"device: {torch.cuda.get_device_name(device)} sm_{capability[0]}{capability[1]} "
        f"({torch.cuda.get_device_properties(device).multi_processor_count} SMs)"
    )
    _log(f"grid: {len(rates)} rates x {len(k_axis)} k x {len(n_axis)} n = {len(rates) * len(k_axis) * len(n_axis)} cells")
    if args.note:
        _log(f"note: {args.note}")

    measured: dict[str, dict[str, list[int]]] = {str(rate): {} for rate in rates}
    raw: dict[str, dict[str, dict]] = {str(rate): {} for rate in rates}
    skipped: list[str] = []

    if args.resume and args.emit_json:
        previous = Path(args.emit_json)
        if previous.is_file():
            payload = json.loads(previous.read_text())
            for rate_text, cells in payload.get("rates", {}).items():
                rate = int(rate_text)
                if rate not in rates:
                    continue
                for key, value in cells.items():
                    size_k, size_n = (int(part) for part in key.split(","))
                    if (rate, size_k, size_n) not in {(rate, k, n) for k in k_axis for n in n_axis}:
                        continue
                    measured[str(rate)][key] = [int(value[0]), int(value[1])]
            _log(f"resumed {sum(len(cells) for cells in measured.values())} cells from {previous}")

    def emit() -> None:
        if not args.emit_json:
            return
        payload = _emit_payload(
            args, measured, skipped, rows_axis, k_axis, n_axis, rates, sms_caps, started
        )
        Path(args.emit_json).write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")

    done = 0
    total = len(rates) * len(k_axis) * len(n_axis)

    for rate in rates:
        for k in k_axis:
            for n in n_axis:
                done += 1
                key = f"{k},{n}"
                existing = measured[str(rate)].get(key)
                if existing is not None and not args.refresh_shapes:
                    continue
                weight_bytes = k * n * rate / 8
                if weight_bytes > args.max_weight_gb * (1 << 30):
                    skipped.append(f"{rate}:{k}x{n}:weight")
                    continue
                if not any(_compatible(shape, k, n) for shape in shapes):
                    skipped.append(f"{rate}:{k}x{n}:no-shape")
                    continue
                per_row = {}
                for rows in rows_axis:
                    per_row[rows] = _sweep_cell(
                        op, pinned, k, n, rows, rate, shapes, args.warmup, args.reps
                    )
                raw[str(rate)][key] = {str(rows): table for rows, table in per_row.items()}
                # The packed table has no m axis.  For fused dispatch the shape
                # is only read while size_m <= 16, so m=16 decides it.
                reference = per_row.get(16) or per_row[max(per_row)]
                if not reference:
                    skipped.append(f"{rate}:{k}x{n}:no-timing")
                    continue
                best = _pick_shape(per_row, args.tie_tolerance)
                if existing is not None:
                    # Shape-only refresh: the num_sms cap came from the original
                    # run, whose timings also scored it, and is kept.
                    measured[str(rate)][key] = [best, existing[1]]
                elif sms_caps:
                    sms = {}
                    for rows in sms_rows:
                        # Fused dispatch replaces the table shape with a row tile
                        # above 16 rows, so the cap has to be scored on the shape
                        # production would actually launch.
                        row_tile = _row_shape(rows) or best
                        if not _compatible(row_tile, k, n):
                            continue
                        sms[rows] = _sweep_sms(
                            op, pinned, k, n, rows, rate, row_tile, sms_caps, args.warmup, args.reps
                        )
                    # A single cap has to serve every batch size, so minimise the
                    # worst slowdown against each row's own best cap.  A cap that
                    # did not launch on every row cannot be scored.
                    scores = {}
                    for cap in sms_caps:
                        penalty = 1.0
                        complete = True
                        for table in sms.values():
                            if not table or cap not in table:
                                complete = False
                                break
                            penalty = max(penalty, table[cap] / min(table.values()))
                        if complete:
                            scores[cap] = penalty
                    if not scores:
                        skipped.append(f"{rate}:{k}x{n}:sms")
                        continue
                    cap = min(scores, key=scores.get)
                    measured[str(rate)][key] = [best, int(cap)]
                    raw[str(rate)][key]["sms"] = {
                        str(rows): {str(key): value for key, value in table.items()} for rows, table in sms.items()
                    }
                else:
                    measured[str(rate)][key] = [best, 0x54]
                # Release the per-cell trellis and activations before the next
                # cell; the biggest grid cells are multi-GB.
                del per_row
                torch.cuda.empty_cache()
                emit()
                if args.progress and done % args.progress == 0:
                    _log(f"  [{done}/{total}] rate={Q4NX_STEP_BITS[rate]} k={k} n={n} -> {measured[str(rate)][key]}")

    payload = _emit_payload(args, measured, skipped, rows_axis, k_axis, n_axis, rates, sms_caps, started)
    emit()
    measured_cells = payload["measured_cells"]
    _log(
        f"\nmeasured {measured_cells}/{payload['grid_cells']} cells, "
        f"{len(payload['fallback'])} explicit fallbacks, {len(skipped)} skipped attempts, "
        f"{time.monotonic() - started_monotonic:.1f}s"
    )
    if args.emit_json:
        _log(f"wrote {args.emit_json}")
    if args.raw_json:
        Path(args.raw_json).write_text(json.dumps(raw, indent=1, sort_keys=True) + "\n")
        _log(f"wrote {args.raw_json}")
    for entry in payload["fallback"]:
        _log(f"  fallback {entry}")

    for rate in rates:
        _log(f"\nrate {Q4NX_STEP_BITS[rate]} bpw (step_bits={rate})")
        _log("  k\\n " + " ".join(f"{n:>6}" for n in n_axis))
        for k in k_axis:
            cells = []
            for n in n_axis:
                entry = measured[str(rate)].get(f"{k},{n}")
                cells.append("  --  " if entry is None else f"{entry[0]}/{entry[1]:<3}")
            _log(f"  {k:>5} " + " ".join(f"{cell:>6}" for cell in cells))
    return 0


def cmd_win(args) -> int:
    device = _require_cuda()
    _prewarm()
    op = torch.ops.gptqmodel_exllamav3.bc_linear_exl3_run_configured
    old = HeaderTable.from_path(Path(args.old_header))
    library = _sweep_library(args.sweep_lib)
    pinned = PinnedConfig(library)
    rates = [int(rate) for rate in args.rates.split(",")]
    rows_axis = [int(value) for value in args.m_axis.split(",")]
    pairs = [tuple(int(value) for value in item.split("x")) for item in args.pairs.split(",")]

    _log(f"device: {torch.cuda.get_device_name(device)}")
    _log("| rate | k | n | m | auto new ms | shape new | auto old ms | shape old | speedup |")
    _log("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    ratios = []
    for rate in rates:
        for k, n in pairs:
            weights = _packed_projection(k, n, rate)
            for rows in rows_axis:
                x = _input(rows, k)
                y = torch.empty((rows, n), device=x.device, dtype=torch.float32)
                pinned.clear()
                new_shape = _call_gemm(op, weights, rows, k, n, rate, x, y)
                old_shape, old_cap = _old_choice(old, 1, rate, k, n)
                # Fused dispatch replaces the table shape with a row tile above
                # 16 rows, so the old side has to pin that same tile; pinning
                # the table shape there would compare against a kernel the old
                # dispatcher never launched.
                old_shape = _row_shape(rows) or old_shape
                if not _compatible(old_shape, k, n):
                    old_shape = _old_choice(old, 1, rate, k, n)[0]
                new_samples, old_samples = [], []
                for _ in range(args.rounds):
                    pinned.clear()
                    new_samples.append(
                        _time_ms(lambda: _call_gemm(op, weights, rows, k, n, rate, x, y), args.warmup, args.reps)
                    )
                    pinned.set(old_shape, old_cap)
                    old_samples.append(
                        _time_ms(lambda: _call_gemm(op, weights, rows, k, n, rate, x, y), args.warmup, args.reps)
                    )
                pinned.clear()
                # Rounds alternate the two configurations, so a slow clock ramp
                # cannot favour whichever side ran second.
                new_ms = statistics.median(new_samples)
                old_ms = statistics.median(old_samples)
                speedup = old_ms / new_ms
                ratios.append(speedup)
                _log(
                    f"| {Q4NX_STEP_BITS[rate]} | {k} | {n} | {rows} | {new_ms:.3f} | {new_shape} | "
                    f"{old_ms:.3f} | {old_shape} | {speedup:.3f}x |"
                )
    if ratios:
        _log(f"\ngeomean speedup: {statistics.geometric_mean(ratios):.3f}x over {len(ratios)} cells")
    return 0


def cmd_verify(args) -> int:
    device = _require_cuda()
    _prewarm()
    from tests.test_exllamav3_torch_padding import _oracle_weight

    op = torch.ops.gptqmodel_exllamav3.bc_linear_exl3_run_configured
    library = _sweep_library(args.sweep_lib)
    pinned = PinnedConfig(library)
    failures = 0
    cases = [tuple(int(value) for value in item.split("x")) for item in args.pairs.split(",")]
    for rate in (int(value) for value in args.rates.split(",")):
        for k, n in cases:
            weights = _packed_projection(k, n, rate)
            trellis, su, sv = weights
            dense = _oracle_weight({"trellis": trellis.cpu(), "suh": su.cpu(), "svh": sv.cpu(), "q4nx": True}, device=device)
            for rows in (int(value) for value in args.m_axis.split(",")):
                x = _input(rows, k)
                expected = (x.float() @ dense + 0.0).float()
                for shape in (int(value) for value in args.shapes.split(",")):
                    if not _compatible(shape, k, n):
                        continue
                    y = torch.empty((rows, n), device=x.device, dtype=torch.float32)
                    pinned.set(shape, min(_effective_num_sms(shape, k, n, 0xFFFF), 0xFF))
                    _call_gemm(op, weights, rows, k, n, rate, x, y)
                    pinned.clear()
                    torch.cuda.synchronize()
                    error = (y - expected).abs().max().item()
                    scale = expected.abs().max().item()
                    ok = error <= args.atol + args.rtol * scale
                    failures += 0 if ok else 1
                    _log(
                        f"rate={Q4NX_STEP_BITS[rate]} k={k} n={n} m={rows} shape={shape} "
                        f"max_abs={error:.3e} scale={scale:.3e} {'ok' if ok else 'FAIL'}"
                    )
    return 0 if failures == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(target: argparse.ArgumentParser) -> None:
        target.add_argument("--rates", default="10,12,14", help="Q4NX step bits (10=2.5, 12=3.0, 14=3.5 bpw)")
        target.add_argument("--warmup", type=int, default=5)
        target.add_argument("--reps", type=int, default=20)
        target.add_argument("--sweep-lib", default=None, help="path to the built extension .so")
        target.add_argument("--shapes", default="1,2,3,4,5,6,7")

    sweep = sub.add_parser("sweep", help="measure every shape per table cell")
    add_common(sweep)
    sweep.add_argument("--header", default=str(PACKED_HEADER), help="header whose axes define the grid")
    sweep.add_argument(
        "--m-axis",
        default="1,16",
        help="batch sizes to measure; the shape minimises the worst slowdown over these rows",
    )
    sweep.add_argument("--sms-sweep", default="8,16,24,32,48,64,80,96,112,128,160,192,255")
    sweep.add_argument("--sms-m-axis", default="16,2048")
    sweep.add_argument("--k-axis", default=None, help="restrict to these k values")
    sweep.add_argument("--n-axis", default=None, help="restrict to these n values")
    sweep.add_argument("--max-weight-gb", type=float, default=2.0)
    sweep.add_argument("--allow-other-cc", action="store_true")
    sweep.add_argument("--emit-json", default=None)
    sweep.add_argument("--raw-json", default=None)
    sweep.add_argument("--resume", action="store_true", help="reuse cells already present in --emit-json")
    sweep.add_argument(
        "--refresh-shapes",
        action="store_true",
        help="re-measure only the shape choice of cells already present in --emit-json",
    )
    sweep.add_argument(
        "--tie-tolerance",
        type=float,
        default=0.03,
        help="prefer a 16-row tile shape within this fraction of the best worst-case score",
    )
    sweep.add_argument("--note", default="", help="free-form note stored with the measurements")
    sweep.add_argument("--progress", type=int, default=25)
    sweep.set_defaults(func=cmd_sweep)

    win = sub.add_parser("win", help="compare auto dispatch against the old table")
    add_common(win)
    win.add_argument("--old-header", required=True, help="packed header from before the retune")
    win.add_argument("--m-axis", default=",".join(str(value) for value in WIN_ROWS))
    win.add_argument("--pairs", default=",".join(f"{k}x{n}" for k, n in WIN_SHAPES))
    win.add_argument("--rounds", type=int, default=1, help="alternate new and old timing rounds; median of rounds")
    win.set_defaults(func=cmd_win)

    verify = sub.add_parser("verify", help="check Q4NX numerics against reconstruct + hgemm")
    add_common(verify)
    verify.add_argument("--m-axis", default="1,16,64")
    verify.add_argument("--pairs", default="512x256,2048x2048,4096x4096")
    verify.add_argument("--atol", type=float, default=2e-3)
    verify.add_argument("--rtol", type=float, default=2e-3)
    verify.set_defaults(func=cmd_verify)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
