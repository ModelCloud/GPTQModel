# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Guards for the measured Ada Q4NX grid in the packed EXL3 kernel map.

``samples_*_q4nx`` is a ``[cc][q4nx rate][k][n]`` grid filled from
``scripts/exl3_kernel_map_q4nx_ada.json``, which
``tests/benchmark_exl3_q4nx_tuning_ada.py`` measures on an RTX 4090.  These
tests pin the three properties that make the table trustworthy:

* the checked-in header is exactly what the generator emits from the JSON,
* every Ada cell is either a measurement or an explicitly declared fallback,
* the dispatcher reads the Q4NX grid (not the bitrate grid) for CC_ADA and
  reproduces the declared fallback when a cell was not measured.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tests.accelerator_gates import requires_nvidia_cuda


REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATOR = REPO_ROOT / "scripts/generate_exl3_kernel_map_packed.py"
SAMPLES = REPO_ROOT / "scripts/exl3_kernel_map_q4nx_ada.json"
PACKED_HEADER = REPO_ROOT / "gptqmodel_ext/exllamav3/quant/exl3_kernel_map_packed.cuh"

Q4NX_STEP_BITS = (10, 12, 14)
Q4NX_BPW = {10: 2.5, 12: 3.0, 14: 3.5}
# cc_values is (2, 3, 4) and 3 is CC_ADA.
CC_ADA_INDEX = 1
BLOCK_MODULI = (128, 256, 512)


class PackedHeader:
    """Minimal reader for the generated header, mirroring the C++ lookups."""

    _U16 = re.compile(r"constexpr uint16_t samples_(\d+)(_q4nx)?\[\] = \{(.*?)\n\};", re.S)
    _INT = re.compile(
        r"constexpr int (cc_values|k_axis|n_axis_128|n_axis_256|n_axis_512|q4nx_rate_step_bits)\[\] = \{(.*?)\n\};",
        re.S,
    )

    def __init__(self, text: str) -> None:
        self.axes = {
            name: [int(value) for value in body.replace("\n", " ").split(",") if value.strip()]
            for name, body in self._INT.findall(text)
        }
        self.samples: dict[int, list[int]] = {}
        self.q4nx: dict[int, list[int]] = {}
        for mod, q4nx, body in self._U16.findall(text):
            payload = [int(value, 16) for value in body.replace("\n", " ").split(",") if value.strip()]
            (self.q4nx if q4nx else self.samples)[int(mod)] = payload
        if set(self.samples) != set(BLOCK_MODULI) or set(self.q4nx) != set(BLOCK_MODULI):
            raise AssertionError("packed header is missing a legacy or Q4NX block")
        if self.axes["q4nx_rate_step_bits"] != list(Q4NX_STEP_BITS):
            raise AssertionError(f"unexpected Q4NX rate axis: {self.axes['q4nx_rate_step_bits']}")
        self.k_axis = self.axes["k_axis"]

    @classmethod
    def from_path(cls, path: Path = PACKED_HEADER) -> "PackedHeader":
        return cls(Path(path).read_text())

    def _axis(self, size_n: int) -> tuple[int, list[int]]:
        for mod in (512, 256):
            if size_n % mod == 0:
                return mod, self.axes[f"n_axis_{mod}"]
        return 128, self.axes["n_axis_128"]

    def legacy_cell(self, step_bits: int, size_k: int, size_n: int) -> tuple[int, int]:
        """The sample the pre-Q4NX dispatcher used: keyed on bits = K + half_k."""
        mod, n_axis = self._axis(size_n)
        bits = step_bits // 4 + int(step_bits % 4 == 2)
        flat = (
            (((CC_ADA_INDEX * 8) + (bits - 1)) * len(self.k_axis)) + _nearest(self.k_axis, size_k)
        ) * len(n_axis) + _nearest(n_axis, size_n)
        packed = self.samples[mod][flat]
        return packed >> 8, packed & 0xFF

    def q4nx_cell(self, step_bits: int, size_k: int, size_n: int) -> tuple[int, int]:
        mod, n_axis = self._axis(size_n)
        rate_idx = Q4NX_STEP_BITS.index(step_bits)
        flat = (
            (((CC_ADA_INDEX * len(Q4NX_STEP_BITS)) + rate_idx) * len(self.k_axis))
            + _nearest(self.k_axis, size_k)
        ) * len(n_axis) + _nearest(n_axis, size_n)
        packed = self.q4nx[mod][flat]
        return packed >> 8, packed & 0xFF


def _nearest(axis: list[int], value: int) -> int:
    best_idx = 0
    best_dist = abs(value - axis[0])
    for idx in range(1, len(axis)):
        dist = abs(value - axis[idx])
        if dist < best_dist:
            best_idx, best_dist = idx, dist
    return best_idx


def _samples() -> dict:
    return json.loads(SAMPLES.read_text())


def _measured_cells(payload: dict) -> dict[tuple[int, int, int], tuple[int, int]]:
    return {
        (int(rate), *[int(part) for part in key.split(",")]): (int(value[0]), int(value[1]))
        for rate, cells in payload["rates"].items()
        for key, value in cells.items()
    }


def _fallback_cells(payload: dict) -> set[tuple[int, int, int]]:
    return {tuple(int(part) for part in item.replace(":", ",").split(",")) for item in payload["fallback"]}


def _grid_cells(payload: dict) -> set[tuple[int, int, int]]:
    return {
        (rate, size_k, size_n)
        for rate in (int(value) for value in payload["axes"]["rates"])
        for size_k in payload["axes"]["k"]
        for size_n in payload["axes"]["n"]
    }


def _run_generator(output: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(GENERATOR), "--output", str(output), *extra],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.timeout(180)
def test_checked_in_q4nx_header_is_regenerated_from_the_ada_json(tmp_path: Path):
    generated = tmp_path / "exl3_kernel_map_packed.cuh"
    result = _run_generator(generated)

    assert result.returncode == 0, result.stderr or result.stdout
    assert "Q4NX Ada cells:" in result.stdout
    assert generated.read_text() == PACKED_HEADER.read_text()


@pytest.mark.timeout(180)
def test_generator_rejects_a_missing_cell_that_is_not_declared_fallback(tmp_path: Path):
    payload = _samples()
    rate, cells = next((rate, cells) for rate, cells in payload["rates"].items() if cells)
    key = next(iter(cells))
    del cells[key]
    undeclared = tmp_path / "undeclared.json"
    undeclared.write_text(json.dumps(payload))
    result = _run_generator(tmp_path / "header.cuh", "--q4nx-samples", str(undeclared))
    assert result.returncode != 0
    assert "neither measured nor declared fallback" in (result.stderr or result.stdout)

    payload["fallback"].append(f"{rate}:{key}")
    declared = tmp_path / "declared.json"
    declared.write_text(json.dumps(payload))
    result = _run_generator(tmp_path / "header.cuh", "--q4nx-samples", str(declared))
    assert result.returncode == 0, result.stderr or result.stdout
    header = PackedHeader.from_path(tmp_path / "header.cuh")
    size_k, size_n = (int(part) for part in key.split(","))
    assert header.q4nx_cell(int(rate), size_k, size_n) == header.legacy_cell(int(rate), size_k, size_n)


def test_q4nx_ada_json_covers_every_cell_explicitly():
    payload = _samples()
    header = PackedHeader.from_path()
    measured = _measured_cells(payload)
    fallback = _fallback_cells(payload)

    assert tuple(payload["capability"]) == (8, 9)
    assert "4090" in payload["device"]
    assert payload["axes"]["k"] == header.k_axis
    assert set(payload["axes"]["n"]) == set(header.axes["n_axis_128"]) | set(header.axes["n_axis_256"]) | set(
        header.axes["n_axis_512"]
    )
    assert measured.keys() | fallback == _grid_cells(payload)
    assert not measured.keys() & fallback
    assert payload["measured_cells"] == len(measured)
    assert payload["grid_cells"] == len(measured) + len(fallback)


def test_q4nx_ada_header_cells_are_the_measured_samples():
    header = PackedHeader.from_path()
    measured = _measured_cells(_samples())

    assert measured
    for (rate, size_k, size_n), cell in measured.items():
        assert header.q4nx_cell(rate, size_k, size_n) == cell, (rate, size_k, size_n)


def test_q4nx_ada_declared_fallbacks_keep_the_bitrate_sample():
    header = PackedHeader.from_path()
    payload = _samples()

    for rate, size_k, size_n in sorted(_fallback_cells(payload)):
        assert header.q4nx_cell(rate, size_k, size_n) == header.legacy_cell(rate, size_k, size_n), (
            rate,
            size_k,
            size_n,
        )


def _packed_projection(size_k: int, size_n: int, step_bits: int, seed: int = 37):
    generator = torch.Generator().manual_seed(seed)
    trellis = torch.randint(
        -32768, 32768, (size_k // 16, size_n // 16, 4 * step_bits), generator=generator, dtype=torch.int16
    ).cuda()
    su = ((2 * torch.randint(0, 2, (size_k,), generator=generator) - 1) * 0.5).half().cuda()
    sv = (torch.randn(size_n, generator=generator) * 0.03).half().cuda()
    return trellis, su, sv


def _dispatch_shape(op, step_bits: int, size_k: int, size_n: int) -> int:
    """Ask the public op which shape it dispatches for a 16-row fused call."""

    trellis, su, sv = _packed_projection(size_k, size_n, step_bits)
    x = (torch.randn((16, size_k), generator=torch.Generator().manual_seed(19)) * 0.1).half().cuda()
    y = torch.empty((16, size_n), device=x.device, dtype=torch.float32)
    shape = op(trellis, su, sv, Q4NX_BPW[step_bits], None, False, False, x, y, False, 0, True, q4nx=True)
    torch.cuda.synchronize()
    return int(shape)


@pytest.fixture(scope="module")
def q4nx_op():
    if not requires_nvidia_cuda():
        pytest.skip("requires NVIDIA CUDA")
    if torch.cuda.get_device_capability() != (8, 9):
        pytest.skip("the Ada Q4NX grid is only exercised on sm_89")
    from gptqmodel.exllamav3.ext import prewarm_exllamav3_extension

    assert prewarm_exllamav3_extension()
    return torch.ops.gptqmodel_exllamav3.bc_linear_exl3_run_configured


@pytest.mark.cuda
def test_q4nx_ada_dispatch_reads_the_q4nx_grid(q4nx_op):
    header = PackedHeader.from_path()
    measured = _measured_cells(_samples())
    differing = sorted(
        cell
        for cell, (shape_idx, _) in measured.items()
        if header.legacy_cell(*cell)[0] != shape_idx
    )
    assert differing, "no measured cell differs from the bitrate grid; the test cannot tell the grids apart"

    # Keep the probe cheap: the table is per (rate, k, n), and a mismatch on a
    # small cell proves the grid is read just as well as on a multi-GB one.
    differing = [cell for cell in differing if cell[1] * cell[2] <= 4 * 1024 * 1024]
    assert differing, "every differing cell is too large to probe; widen the sweep"

    step = max(1, len(differing) // 6)
    for rate, size_k, size_n in differing[::step][:6]:
        assert _dispatch_shape(q4nx_op, rate, size_k, size_n) == measured[(rate, size_k, size_n)][0], (
            rate,
            size_k,
            size_n,
        )


@pytest.mark.cuda
def test_q4nx_ada_dispatch_falls_back_to_the_bitrate_grid(q4nx_op):
    header = PackedHeader.from_path()
    fallback = sorted(_fallback_cells(_samples()))
    if not fallback:
        pytest.skip("the sweep measured every Ada Q4NX cell")
    small = [cell for cell in fallback if cell[1] * cell[2] <= 4 * 1024 * 1024]
    if not small:
        pytest.skip("every declared fallback is too large to probe")

    for rate, size_k, size_n in small[:4]:
        assert _dispatch_shape(q4nx_op, rate, size_k, size_n) == header.legacy_cell(rate, size_k, size_n)[0], (
            rate,
            size_k,
            size_n,
        )
