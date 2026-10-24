# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# adapted from @qwopqwop200 's [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa/tree/cuda), which itself is based on [gptq](https://github.com/IST-DASLab/gptq)

import torch
import torch.nn as nn

from ..utils.logger import setup_logger
from .config import BaseQuantizeConfig, MSEConfig, _normalize_quant_bits, resolve_quant_format


log = setup_logger()

HF_OPTIMUM = "hf_optimum"


def _replace_zero_scale(
    scale: torch.Tensor,
    zero: torch.Tensor | None = None,
    *,
    zero_mask: torch.Tensor | None = None,
):
    """Replace exact zero scales with a representable positive value.

    ``1e-8`` is not representable as a normal ``float16`` value and would
    underflow to zero there.  Use the dtype's smallest normal instead so the
    division below remains finite for every supported weight dtype.
    """

    if zero_mask is None:
        zero_mask = scale == 0
    safe_scale = torch.where(
        zero_mask,
        torch.full_like(scale, torch.finfo(scale.dtype).tiny),
        scale,
    )
    if zero is None:
        return safe_scale

    # A zero scale can also make the affine zero point non-finite.  Preserve
    # valid zero points, but pin undefined lanes to zero so the result stays
    # finite.
    undefined_zero = zero_mask & ~torch.isfinite(zero)
    safe_zero = torch.where(undefined_zero, torch.zeros_like(zero), zero)
    return safe_scale, safe_zero


def _neutralize_dead_columns(
    hessian: torch.Tensor,
    weight: torch.Tensor,
    *column_tensors: torch.Tensor,
) -> None:
    """Neutralize calibration-dead input channels in place.

    A channel with a zero Hessian diagonal carries no calibration signal. Its
    diagonal entry is replaced with one so the Cholesky inverse stays
    well-defined, and the matching weight (and optional auxiliary) columns are
    zeroed.

    Boolean-mask assignment (``H[dead, dead] = 1`` / ``W[:, dead] = 0``)
    performs a ``nonzero`` scan and a device-to-host synchronization per
    module. The ``masked_fill_`` forms below are equivalent and stay on the
    device.
    """

    dead = torch.diagonal(hessian) == 0
    hessian.diagonal().masked_fill_(dead, 1.0)
    weight.masked_fill_(dead.unsqueeze(0), 0.0)
    for tensor in column_tensors:
        tensor.masked_fill_(dead.unsqueeze(0), 0.0)


def quantize(x, scale, zero, maxq, requires_groupwise_processing: bool):
    if maxq < 0:
        return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero

    scale, zero = _replace_zero_scale(scale, zero)

    if requires_groupwise_processing:
        q = torch.clamp(torch.round(x / scale), -maxq, maxq)
        return scale * q
    else:
        q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
        return scale * (q - zero)


def _mse_shrink_grid(
    x: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    mse: float,
    *,
    maxq_value: int,
    groupwise: bool,
    guard_zero_scale: bool,
    scale: torch.Tensor,
    zero: torch.Tensor,
    grid_budget: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick each channel's best ``(scale, zero)`` candidate by ``|deq(x)-x|^mse``.

    ``scales``/``zeros`` list every ``(n_candidates, rows)`` candidate in
    priority order.  The per-step loop previously launched a dozen small
    kernels per candidate; batching the grid collapses that into a few large
    ops per chunk.  Candidate order, objective values, and the first-minimum
    tie break are identical to evaluating candidates one at a time, so the
    returned ``scale``/``zero`` are bitwise-equal to the sequential loop's
    result.  ``scale``/``zero`` are the current winners the candidates must
    beat.
    """
    dev = x.device
    rows, cols = x.shape
    n_candidates = scales.shape[0]

    # The elementwise chain promotes exactly like the per-candidate
    # ``quantize()`` call did; a single up-front cast keeps it that way.
    work_dtype = torch.result_type(x, scales)
    scales = scales.to(work_dtype)
    zeros = zeros.to(work_dtype)
    # The affine path needs only the ``q`` workspace; trits additionally
    # hold two f32 dequant masks while forming the output.
    elem_size = torch.empty((), dtype=work_dtype).element_size()
    bytes_per_elem = (
        elem_size
        if maxq_value >= 0
        else 2 * torch.tensor(0.0).element_size() + max(elem_size, 4)
    )

    # Per-chunk workspace cap in bytes. On GPU a large chunk amortizes the
    # per-candidate launch storm and feeds the wide memory bus; on CPU the same
    # footprint only evicts the L3-resident working set the sequential loop
    # enjoyed, so ``None`` resolves to a small CPU default.
    budget = MSEConfig.resolve_budget_bytes(grid_budget, dev)
    chunk = max(
        1, min(n_candidates, budget // max(bytes_per_elem * rows * cols, 1))
    )

    # ``x`` can be a transposed view (activation find_params reshapes 2-D/3-D
    # inputs with ``.t()``).  The sequential code's ``quantize`` output then
    # inherited that column-major physical order, and its ``torch.sum(q, 1)``
    # accumulated in that order.  Keep the workspace in the same physical
    # order so the reduction walks memory identically: a 1-ULP different sum
    # would flip strict-``<`` ties and change candidate selection.
    col_major = 0 < x.stride(0) < x.stride(1)
    if col_major:
        q_phys = torch.empty((chunk, cols, rows), dtype=work_dtype, device=dev)
        q_buf = q_phys.transpose(1, 2)  # logical (n, rows, cols)
    else:
        q_phys = torch.empty((chunk, rows, cols), dtype=work_dtype, device=dev)
        q_buf = q_phys
    best = torch.full([rows], float("inf"), device=dev)
    xr = x.unsqueeze(0)
    for start in range(0, n_candidates, chunk):
        n = min(chunk, n_candidates - start)
        sc = scales[start:start + n]
        zc = zeros[start:start + n]
        q = q_buf[:n]

        if maxq_value < 0:
            # ``quantize`` bypasses the zero-scale guard for ``maxq < 0``;
            # keep the same mask -> dequant -> error order for trits.
            s3, z3 = sc.unsqueeze(-1), zc.unsqueeze(-1)
            q.copy_((xr > s3 / 2).float() * s3 + (xr < z3 / 2).float() * z3)
        else:
            if guard_zero_scale:
                # Selection must store the raw candidates, so keep ``sc``/``zc``
                # untouched and only guard the tensors the division reads.
                safe_scale, safe_zero = _replace_zero_scale(sc, zc)
            else:
                safe_scale, safe_zero = sc, zc
            s3, z3 = safe_scale.unsqueeze(-1), safe_zero.unsqueeze(-1)
            torch.div(xr, s3, out=q)
            q.round_()
            if groupwise:
                q.clamp_(-maxq_value, maxq_value)
                q.mul_(s3)
            else:
                # Same op order as ``quantize``: +zero, clamp, -zero, *scale.
                q.add_(z3).clamp_(0, maxq_value).sub_(z3).mul_(s3)
        q.sub_(xr)
        q.abs_()
        q.pow_(mse)
        err = (
            q_phys[:n].sum(dim=1)
            if col_major
            else q.reshape(n * rows, cols).sum(dim=1).reshape(n, rows)
        )

        # NaN objectives could never satisfy the sequential ``err < best``;
        # pin them to +inf so ``argmin`` cannot select them either.
        err = torch.where(
            torch.isnan(err), torch.full_like(err, float("inf")), err
        )
        idx = err.argmin(dim=0)  # first minimum == sequential tie order
        chunk_best = err.gather(0, idx.unsqueeze(0)).squeeze(0)
        better = chunk_best < best  # strict across chunks
        scale = torch.where(
            better, sc.gather(0, idx.unsqueeze(0)).squeeze(0), scale
        )
        zero = torch.where(
            better, zc.gather(0, idx.unsqueeze(0)).squeeze(0), zero
        )
        best = torch.where(better, chunk_best, best)

    return scale, zero


class Quantizer(nn.Module):
    def __init__(self, qcfg: BaseQuantizeConfig, shape=1, name: str=None):
        super(Quantizer, self).__init__()

        self.qcfg = qcfg
        self.register_buffer("maxq", torch.tensor(0))
        # Host mirror of the ``maxq`` buffer.  ``configure()`` is the only writer
        # of ``maxq``, so the mirror cannot drift, and reading the bound as a
        # Python int keeps the block loop free of device-to-host synchronizes.
        self._maxq_value = 0
        self.register_buffer("scale", torch.zeros(shape))
        self.register_buffer("zero", torch.zeros(shape))

        self.name=name

    def requires_groupwise_processing(self) -> bool:
        return False

    def _search_adjacent_zero_points(self) -> bool:
        """Return whether range search should optimize the 2-bit affine orientation."""

        return (
            self._maxq_value == 3
            and not self.qcfg.sym
            and not self.requires_groupwise_processing()
        )

    @staticmethod
    def _adjacent_zero_point(
        xmin: torch.Tensor,
        scale: torch.Tensor,
        primary: torch.Tensor,
        *,
        maxq: int,
    ) -> torch.Tensor:
        """Return the other integer neighboring the ideal affine zero point."""

        ideal = -xmin / scale
        lower = torch.floor(ideal).clamp_(0, maxq)
        upper = torch.ceil(ideal).clamp_(0, maxq)
        return torch.where(primary == lower, upper, lower)

    # FIXME, optimum shouldn't call this directly, it should call hf_configure
    def configure(
        self,
        perchannel=False,
        grid=100,
        maxshrink=0.8,
        trits=False,
        bits: int | str | None = None, # for hf compat
        sym: bool | None = None, # for hf compat
    ):
        if self.name == HF_OPTIMUM:
            if bits is not None:
                self.qcfg.bits = _normalize_quant_bits(bits, format_value=resolve_quant_format(self.qcfg.format, self.qcfg.method))
            if sym is not None:
                self.qcfg.sym = sym

        if self.requires_groupwise_processing():
            self.maxq = torch.tensor(2 ** (self.qcfg.bits - 1) - 1)
        else:
            self.maxq = torch.tensor(2 ** self.qcfg.bits - 1)

        self.perchannel = perchannel
        self.grid = grid
        self.maxshrink = maxshrink
        if trits:
            self.maxq = torch.tensor(-1)
        self._maxq_value = int(self.maxq)

    def find_params(self, x, weight=False):
        dev = x.device
        if self.maxq.device != dev:
            self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            if weight:
                x = x.flatten(1)
            else:
                if len(shape) == 4:
                    x = x.permute([1, 0, 2, 3])
                    x = x.flatten(1)
                if len(shape) == 3:
                    x = x.reshape((-1, shape[-1])).t()
                if len(shape) == 2:
                    x = x.t()
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        # ``aminmax`` returns both extremes from a single reduction pass; the
        # separate ``min(dim=1)``/``max(dim=1)`` pair also materializes the
        # arg-min/arg-max indices that the range search never consumes.
        xmin_raw, xmax_raw = torch.aminmax(x, dim=1)
        xmin = torch.minimum(xmin_raw, tmp)
        xmax = torch.maximum(xmax_raw, tmp)

        if self.qcfg.sym:
            # ``xmax >= |xmin|`` from here on, so the negative mirror is exactly
            # the ``xmin < 0`` lane.  Expressing it as an elementwise select
            # keeps the decision on the device: the masked write it replaces
            # forced a ``nonzero`` scan plus a host round trip per call.
            xmax = torch.maximum(torch.abs(xmin), xmax)
            xmin = torch.where(xmin < 0, -xmax, xmin)
        # Empty groups (all-zero ranges) get the +-1 sentinel so the scale below
        # stays finite.  ``both_zero`` selects the same lanes as the previous
        # pair of masked writes without a device-to-host synchronization.
        # In the symmetric path ``xmax >= |xmin|``, so a zero ``xmax`` already
        # proves the whole range is zero.  The non-symmetric path needs both
        # comparisons because an all-negative group has ``xmax == 0``.
        both_zero = xmax == 0 if self.qcfg.sym else (xmin == 0) & (xmax == 0)
        xmin = torch.where(both_zero, xmin - 1, xmin)
        xmax = torch.where(both_zero, xmax + 1, xmax)

        if self._maxq_value < 0:
            self.scale = xmax
            self.zero = xmin
        else:
            if self.requires_groupwise_processing():
                self.scale = xmax / self.maxq
                self.zero = torch.zeros_like(self.scale)
            else:
                self.scale = (xmax - xmin) / self.maxq
                if self.qcfg.sym:
                    # ``maxq`` is odd here, so the host-side half is exact.
                    self.zero = torch.full_like(self.scale, (self._maxq_value + 1) / 2)
                else:
                    self.zero = torch.round(-xmin / self.scale)

        mse_cfg = getattr(self.qcfg, "mse", None)
        mse = mse_cfg.norm if isinstance(mse_cfg, MSEConfig) else float(mse_cfg or 0.0)
        if mse > 0.0:
            self._mse_shrink_grid(x, xmin, xmax, mse)
        if not self.perchannel:
            if weight:
                tmp = shape[0]
            else:
                tmp = shape[1] if len(shape) != 3 else shape[2]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        if weight:
            shape = [-1] + [1] * (len(shape) - 1)
            self.scale = self.scale.reshape(shape)
            self.zero = self.zero.reshape(shape)
            return
        if len(shape) == 4:
            self.scale = self.scale.reshape((1, -1, 1, 1))
            self.zero = self.zero.reshape((1, -1, 1, 1))
        if len(shape) == 3:
            self.scale = self.scale.reshape((1, 1, -1))
            self.zero = self.zero.reshape((1, 1, -1))
        if len(shape) == 2:
            self.scale = self.scale.unsqueeze(0)
            self.zero = self.zero.unsqueeze(0)

    def _mse_shrink_grid(
        self,
        x: torch.Tensor,
        xmin: torch.Tensor,
        xmax: torch.Tensor,
        mse: float,
    ) -> None:
        """Evaluate every shrink-grid candidate's scale/zero and keep the winners.

        The per-step loop previously launched a dozen small kernels for each of
        the ``maxshrink * grid`` candidates (times two when the 2-bit adjacent
        zero point is searched).  Batching the grid collapses that into a few
        large ops per chunk.  Candidate order, objective values, and the
        first-minimum tie break are identical to evaluating the candidates one
        at a time, so the selected ``scale``/``zero`` are bitwise-equal to the
        sequential loop's result.
        """
        dev = x.device
        rows = x.shape[0]
        steps = int(self.maxshrink * self.grid)
        if steps <= 0:
            return

        groupwise = self.requires_groupwise_processing()
        search_adjacent = self._search_adjacent_zero_points()

        # ``1 - i/grid`` in host float, rounded to ``xmin``'s dtype, reproduces
        # the values the per-step loop got from ``p * xmin``.
        p = torch.tensor(
            [1 - i / self.grid for i in range(steps)],
            dtype=xmin.dtype,
            device=dev,
        ).unsqueeze(1)
        xmin1 = p * xmin.unsqueeze(0)
        xmax1 = p * xmax.unsqueeze(0)
        scale1 = xmax1 / self.maxq if groupwise else (xmax1 - xmin1) / self.maxq
        if self.qcfg.sym:
            zero1 = self.zero.unsqueeze(0).expand(steps, rows)
        else:
            zero1 = torch.round(-xmin1 / scale1)

        # Step-major order -- (i=0,z0), (i=0,z1), (i=1,z0), ... -- preserves the
        # sequential candidate order, so the first-minimum ``argmin`` below
        # picks the same winner the sequential strict ``<`` updates did.
        if search_adjacent:
            alternate_zero = self._adjacent_zero_point(
                xmin1,
                scale1,
                zero1,
                maxq=self._maxq_value,
            )
            scales = scale1.repeat_interleave(2, dim=0)
            zeros = torch.stack((zero1, alternate_zero), dim=1).reshape(2 * steps, rows)
        else:
            scales, zeros = scale1, zero1

        mse_cfg = getattr(self.qcfg, "mse", None)
        self.scale, self.zero = _mse_shrink_grid(
            x,
            scales,
            zeros,
            mse,
            maxq_value=self._maxq_value,
            groupwise=groupwise,
            guard_zero_scale=True,
            scale=self.scale,
            zero=self.zero,
            grid_budget=mse_cfg.grid_budget if isinstance(mse_cfg, MSEConfig) else None,
        )

    def quantize(self, x):
        # ``maxq`` is a device tensor, so passing it into the elementwise helper
        # would make its ``maxq < 0`` branch and clamp bounds read the scalar back
        # to the host on every block.  ``_maxq_value`` mirrors it exactly.
        return quantize(x, self.scale, self.zero, self._maxq_value, self.requires_groupwise_processing())

    # def enabled(self):
    #     return self.maxq > 0

    # def ready(self):
    # return torch.all(self.scale != 0)

class QQQQuantizer(Quantizer):
    def requires_groupwise_processing(self) -> bool:
        return self.qcfg.group_size == -1 and self.qcfg.sym

__all__ = ["Quantizer"]
