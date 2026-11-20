# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Input-only W4A floating activation policies used during GPTQ replay."""

from __future__ import annotations

import math
import warnings
from contextlib import nullcontext

import torch


FP8_E4M3_MAX = 448.0
FP4_E2M1_MAX = 6.0
NVFP4_E4M3_SEARCH_RADIUS = 8
NVFP4_RECIPES = {
    "nvidia", "nvidia_headroom", "four_six", "least_squares", "least_squares_headroom", "least_squares_grid"
}
_LEGACY_NVFP4_RECIPE_ALIASES = {
    "lsq": "least_squares",
    "lsq_headroom": "least_squares_headroom",
    "lsq_grid": "least_squares_grid",
}


def normalize_nvfp4_recipe(recipe: str) -> str:
    """Expand legacy checkpoint names; this is fitting, not learned-scale QAT."""
    if not isinstance(recipe, str):
        raise ValueError(f"Unsupported NVFP4 scale recipe: {recipe!r}.")
    recipe = _LEGACY_NVFP4_RECIPE_ALIASES.get(recipe, recipe)
    if recipe not in NVFP4_RECIPES:
        raise ValueError(f"Unsupported NVFP4 scale recipe: {recipe!r}.")
    return recipe


def nvfp4_kernel_recipe(recipe: str) -> str:
    """Return the local block-scale algorithm used by a complete recipe."""
    recipe = normalize_nvfp4_recipe(recipe)
    return {
        "nvidia_headroom": "nvidia",
        "least_squares_headroom": "least_squares",
    }.get(recipe, recipe)


def nvfp4_uses_headroom(recipe: str | None) -> bool:
    if recipe is not None:
        recipe = normalize_nvfp4_recipe(recipe)
    return recipe in {"nvidia_headroom", "least_squares_headroom"}


class NVFP4ActivationHeadroom:
    """Bounded histogram implementation of NVIDIA's activation scale calibration.

    The histogram records one maximum per 16-value input block.  It is small
    and fixed size regardless of calibration token count, so collecting the
    scale does not retain activation tensors or weaken the memory guard.
    Blocks outside the histogram domain keep their exact count, minimum, and
    maximum, so a percentile that resolves into an edge bin still reports a
    real tail magnitude instead of that bin's midpoint.
    """

    _FP8_NORMAL_DYNAMIC_RANGE = 28672.0
    _ANCHOR_FLOOR_RATIO = 1e6

    def __init__(self, *, anchor_percentile: float = 1.0,
                 upper_percentile: float = 99.99, rho: float = 16384.0,
                 num_bins: int = 512, log2_min: float = -40.0,
                 log2_max: float = 40.0):
        if not 0.0 < anchor_percentile <= 100.0:
            raise ValueError("anchor_percentile must be in (0, 100].")
        if not 0.0 < upper_percentile <= 100.0:
            raise ValueError("upper_percentile must be in (0, 100].")
        if not 0.0 < rho < self._FP8_NORMAL_DYNAMIC_RANGE:
            raise ValueError("rho must be in (0, 28672).")
        self.anchor_percentile = float(anchor_percentile)
        self.upper_percentile = float(upper_percentile)
        self.rho = float(rho)
        self.num_bins = int(num_bins)
        self.log2_min = float(log2_min)
        self.log2_max = float(log2_max)
        self.hist: torch.Tensor | None = None
        self.running_max: torch.Tensor | None = None
        self.running_min: torch.Tensor | None = None
        # Exact count/min/max of the blocks outside the histogram domain,
        # ordered underflow then overflow.
        self.tail_count: torch.Tensor | None = None
        self.tail_min: torch.Tensor | None = None
        self.tail_max: torch.Tensor | None = None

    def _bin_index(self, values: torch.Tensor) -> torch.Tensor:
        fraction = (values - self.log2_min) / (self.log2_max - self.log2_min)
        return (fraction * self.num_bins).floor().long().clamp_(0, self.num_bins - 1)

    @torch.no_grad()
    def collect(self, x: torch.Tensor) -> None:
        if x.numel() == 0:
            return
        if x.shape[-1] % 16:
            raise ValueError("NVFP4 headroom calibration requires input K divisible by 16.")
        block_amax = x.detach().abs().reshape(-1, 16).amax(dim=-1).float()
        if not bool(torch.isfinite(block_amax).all()):
            raise ValueError("NVFP4 headroom calibration input contains NaN or infinity.")
        current = block_amax.max()
        self.running_max = current if self.running_max is None else torch.maximum(
            self.running_max.to(current.device), current
        )
        nonzero = block_amax[block_amax > 0]
        if nonzero.numel() == 0:
            return
        current_min = nonzero.min()
        self.running_min = current_min if self.running_min is None else torch.minimum(
            self.running_min.to(current_min.device), current_min
        )
        log2_values = torch.log2(nonzero)
        underflow = log2_values < self.log2_min
        overflow = log2_values >= self.log2_max
        counts = torch.bincount(
            self._bin_index(log2_values[~(underflow | overflow)]), minlength=self.num_bins
        )
        if self.hist is None:
            self.hist = torch.zeros(
                self.num_bins, dtype=torch.int64, device=block_amax.device
            )
            # The edge bins would fold every out-of-domain block into one count
            # whose midpoint is decades away from the data (#5651).
            self.tail_count = torch.zeros(2, dtype=torch.int64, device=self.hist.device)
            self.tail_min = torch.full(
                (2,), float("inf"), dtype=torch.float32, device=self.hist.device
            )
            self.tail_max = torch.zeros(2, dtype=torch.float32, device=self.hist.device)
        self.hist += counts.to(self.hist.device)
        for region, mask in ((0, underflow), (1, overflow)):
            values = nonzero[mask].to(self.hist.device)
            if values.numel():
                self.tail_count[region] += values.numel()
                self.tail_min[region] = torch.minimum(self.tail_min[region], values.min())
                self.tail_max[region] = torch.maximum(self.tail_max[region], values.max())

    def _tail_percentile(self, region: int, fraction: float) -> float:
        """Interpolate a percentile inside the observed under/overflow range."""
        lo = float(self.tail_min[region].item())
        hi = float(self.tail_max[region].item())
        if lo == hi or fraction <= 0.0:
            return lo
        if fraction >= 1.0:
            return hi
        return float(2.0 ** (math.log2(lo) + fraction * (math.log2(hi) - math.log2(lo))))

    def _clamp_to_observed(self, value: float) -> float:
        """A percentile estimate must stay inside the block magnitudes seen."""
        if self.running_min is not None:
            value = max(value, float(self.running_min.item()))
        if self.running_max is not None:
            value = min(value, float(self.running_max.item()))
        return value

    def _percentile(self, percentile: float,
                    floor_value: float | None = None) -> float | None:
        if self.hist is None:
            return None
        counts = self.hist.clone()
        under_count = int(self.tail_count[0].item())
        over_count = int(self.tail_count[1].item())
        if floor_value is not None and floor_value > 0:
            floor = torch.tensor(
                floor_value, device=counts.device, dtype=torch.float32
            )
            floor_bin = int(self._bin_index(torch.log2(floor)).item())
            counts[:floor_bin] = 0
            if floor_bin > 0:
                # The floor sits above the histogram's first edge, so every
                # underflow block is below it.
                under_count = 0
        total = int(counts.sum().item()) + under_count + over_count
        if total == 0:
            return None
        target = percentile / 100.0 * total
        if under_count and target <= under_count:
            return self._tail_percentile(0, target / under_count)
        target -= under_count
        cumulative = counts.cumsum(0)
        in_range_total = int(cumulative[-1].item())
        if over_count and target > in_range_total:
            return self._tail_percentile(1, (target - in_range_total) / over_count)
        index = int(torch.searchsorted(
            cumulative, torch.tensor(target, dtype=torch.float32, device=counts.device)
        ).clamp(
            0, self.num_bins - 1
        ).item())
        exponent = self.log2_min + (index + 0.5) / self.num_bins * (
            self.log2_max - self.log2_min
        )
        return self._clamp_to_observed(2.0 ** exponent)

    @torch.no_grad()
    def compute_amax(self) -> torch.Tensor:
        if self.hist is None or self.running_max is None:
            raise ValueError("NVFP4 headroom calibration observed no nonzero activation blocks.")
        upper = (float(self.running_max.item()) if self.upper_percentile >= 100.0
                 else self._percentile(self.upper_percentile))
        anchor = self._percentile(
            self.anchor_percentile,
            floor_value=upper / self._ANCHOR_FLOOR_RATIO if upper else None,
        )
        if not upper or not anchor:
            return self.running_max.float()
        headroom = self.rho * anchor
        if headroom <= upper:
            warnings.warn(
                "NVFP4 block range leaves no requested activation headroom; "
                "using the calibrated upper percentile.",
                stacklevel=2,
            )
            headroom = upper
        return torch.tensor(
            headroom, device=self.running_max.device, dtype=torch.float32
        )


def fp8_token_qdq(x: torch.Tensor) -> torch.Tensor:
    """Round each Linear input row to E4M3 with its own dynamic scale.

    A finite row always produces a finite result.  ``amax / 448`` can underflow
    to zero while ``amax`` is still a positive FP32 subnormal, and dividing by
    that zero turns the row's zero elements into ``0 / 0`` NaN.  Every value in
    such a row is below E4M3's smallest subnormal, so the unscaled fallback
    still rounds it to zero; the fallback only removes the NaN.
    """
    if x.shape[-1] == 0 or x.numel() == 0:
        return x
    x32 = x.float()
    if not bool(torch.isfinite(x32).all()):
        raise ValueError("W4AFP8 Linear input contains NaN or infinity.")
    amax = x32.abs().amax(dim=-1, keepdim=True)
    scale = amax / FP8_E4M3_MAX
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    rounded = (x32 / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return (rounded.float() * scale).to(x.dtype)


def nvfp4_global_scale(amax: torch.Tensor | float, *, grid_dtype: torch.dtype | None = None,
                       recipe: str = "least_squares") -> torch.Tensor:
    """Return an FP32 global scale, optionally rounded on the source tensor grid.

    Retain a positive FP32 scale when source-grid rounding would underflow it
    to zero, as can happen for small but representable FP16 activations.
    """
    value = torch.as_tensor(amax, dtype=torch.float32)
    if not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
        raise ValueError("NVFP4 activation maximum must be finite and nonnegative.")
    local_recipe = nvfp4_kernel_recipe(recipe)
    # NVIDIA's reference uses the complete E2M1 range. The optimized recipes
    # reserve E4M3 headroom for their additional M=4 candidate.
    denominator = (
        FP4_E2M1_MAX
        if local_recipe == "nvidia" or nvfp4_uses_headroom(recipe)
        else 4.0
    )
    scale = torch.where(value > 0, value / (FP8_E4M3_MAX * denominator), torch.ones_like(value))
    if grid_dtype is not None:
        if grid_dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("NVFP4 global-scale grid must be FP16, BF16, or FP32.")
        rounded = scale.to(grid_dtype).float()
        scale = torch.where(rounded > 0, rounded, scale)
    return scale


def _e2m1_nearest_values(x: torch.Tensor) -> torch.Tensor:
    """E2M1 rounding to nearest, ties to even, with finite saturation."""
    magnitude = x.abs().clamp(max=FP4_E2M1_MAX)
    code = torch.zeros_like(magnitude, dtype=torch.uint8)
    for boundary, code_value, ties_up in (
        (0.25, 1, False), (0.75, 2, True), (1.25, 3, False),
        (1.75, 4, True), (2.5, 5, False), (3.5, 6, True), (5.0, 7, False),
    ):
        code = torch.where(magnitude >= boundary if ties_up else magnitude > boundary, code_value, code)
    values = torch.tensor((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0), device=x.device)
    return values[code.long()].copysign(x)


def _nvfp4_mse_quantize_blocks(
    blocks: torch.Tensor, global_scale: torch.Tensor, recipe: str = "least_squares"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose an E4M3 block scale by monotone least-squares refinement.

    The two Four-Over-Six scales seed separate E2M1 code assignments.  For
    each assignment, solve the scalar least-squares problem, round the result
    back to E4M3, and retain it only when its true block SSE is lower.  One
    final refinement of the winning assignment makes this a strict superset
    of the old two-candidate search without introducing a non-hardware scale.
    """
    recipe = nvfp4_kernel_recipe(recipe)
    block_amax = blocks.abs().amax(dim=-1, keepdim=True)
    # Compare candidates in units of a per-block power of two.  Scaling by
    # 2**-e is exact for finite values, so this keeps the ranking identical to
    # the absolute SSE while preventing FP32 overflow for large activations.
    _, block_exponent = torch.frexp(block_amax)
    block_exponent = block_exponent.to(torch.int32)
    best_error = torch.full_like(block_amax, torch.inf)
    best_local = torch.ones_like(block_amax).to(torch.float8_e4m3fn)
    best_values = torch.zeros_like(blocks)
    seeds = []

    def evaluate(local: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scale = local.float() * global_scale
        values = _e2m1_nearest_values(blocks / scale)
        # ``ldexp`` scales exactly, which ``2**-e`` cannot for subnormal blocks
        # (the power itself can overflow), but it launches on the *current*
        # device and faults when ``blocks`` lives on another GPU (#5263). Enter
        # the tensors' device so the launch targets the right one.
        with torch.cuda.device(blocks.device) if blocks.is_cuda else nullcontext():
            scaled_blocks = torch.ldexp(blocks, -block_exponent)
            scaled_scale = torch.ldexp(scale, -block_exponent)
        reconstructed = values * scaled_scale
        # A zero code reconstructs to exactly zero; avoid 0 * inf when the
        # candidate scale itself overflows.
        reconstructed = torch.where(
            values == 0, torch.zeros_like(reconstructed), reconstructed
        )
        error = (reconstructed - scaled_blocks).square().sum(dim=-1, keepdim=True)
        # A zero or non-finite scale cannot produce a meaningful candidate.
        valid = torch.isfinite(scale) & (scale > 0)
        return values, torch.where(valid, error, torch.full_like(error, torch.inf))

    def require_finite(error: torch.Tensor) -> None:
        if not bool(torch.isfinite(error).all()):
            raise ValueError(
                "NVFP4 block-scale search produced no finite candidate; "
                "the activation magnitude is not representable."
            )

    def refine(values: torch.Tensor) -> torch.Tensor:
        denominator = values.square().sum(dim=-1, keepdim=True)
        optimal = (blocks * values).sum(dim=-1, keepdim=True) / denominator.clamp_min(1.0)
        inverse = optimal / global_scale
        inverse = torch.where(
            denominator > 0,
            inverse.clamp(min=2.0**-9, max=FP8_E4M3_MAX),
            torch.ones_like(inverse),
        )
        return inverse.to(torch.float8_e4m3fn)

    maxima = (FP4_E2M1_MAX,) if recipe == "nvidia" else (4.0, FP4_E2M1_MAX)
    for maximum in maxima:
        inverse = block_amax / (maximum * global_scale)
        inverse = torch.where(
            block_amax > 0,
            inverse.clamp(min=2.0**-9, max=FP8_E4M3_MAX),
            torch.ones_like(inverse),
        )
        local = inverse.to(torch.float8_e4m3fn)
        values, error = evaluate(local)
        seeds.append((local, values))
        use = error < best_error
        best_error = torch.where(use, error, best_error)
        best_local = torch.where(use, local, best_local)
        best_values = torch.where(use, values, best_values)

    if recipe in {"nvidia", "four_six"}:
        require_finite(best_error)
        return best_local, best_values

    for _local, values in seeds:
        local = refine(values)
        values, error = evaluate(local)
        use = error < best_error
        best_error = torch.where(use, error, best_error)
        best_local = torch.where(use, local, best_local)
        best_values = torch.where(use, values, best_values)

    local = refine(best_values)
    values, error = evaluate(local)
    use = error < best_error
    best_error = torch.where(use, error, best_error)
    best_local = torch.where(use, local, best_local)
    best_values = torch.where(use, values, best_values)

    if recipe == "least_squares":
        require_finite(best_error)
        return best_local, best_values

    # Coordinate refinement can stop at a local E2M1 assignment.  Search a
    # small, fixed neighborhood on the actual positive E4M3 bit grid.  E4M3FN
    # encodings 1..126 are monotonically increasing finite values, so this is
    # an exact hardware-scale search with bounded work and no synthetic scale.
    center = best_local.view(torch.uint8).to(torch.int16)
    for offset in range(-NVFP4_E4M3_SEARCH_RADIUS,
                        NVFP4_E4M3_SEARCH_RADIUS + 1):
        candidate = (center + offset).clamp_(1, 126).to(torch.uint8).view(
            torch.float8_e4m3fn
        )
        values, error = evaluate(candidate)
        use = error < best_error
        best_error = torch.where(use, error, best_error)
        best_local = torch.where(use, candidate, best_local)
        best_values = torch.where(use, values, best_values)
    require_finite(best_error)
    return best_local, best_values


def nvfp4_block_qdq(x: torch.Tensor, global_scale: torch.Tensor | float,
                    recipe: str = "least_squares") -> torch.Tensor:
    """Quantize block-16 values with hardware-scale least-squares refinement."""
    if x.numel() == 0:
        return x
    if x.shape[-1] % 16:
        raise ValueError("NVFP4 Linear input K must be divisible by 16.")
    x32 = x.float()
    if not bool(torch.isfinite(x32).all()):
        raise ValueError("NVFP4 Linear input contains NaN or infinity.")
    global_scale = torch.as_tensor(global_scale, device=x.device, dtype=torch.float32)
    if not bool(torch.isfinite(global_scale).all()) or bool((global_scale <= 0).any()):
        raise ValueError("NVFP4 global scale must be positive and finite.")
    blocks = x32.reshape(*x.shape[:-1], x.shape[-1] // 16, 16)
    local, values = _nvfp4_mse_quantize_blocks(blocks, global_scale, recipe)
    return (values * local.float() * global_scale).reshape_as(x).to(x.dtype)
