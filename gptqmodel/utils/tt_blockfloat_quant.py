# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Quantize weights onto a Tenstorrent block-float grid.

Tenstorrent's ``bfloat8_b`` / ``bfloat4_b`` are *storage* formats the unpack
stage expands in the data path, so a weight stored that way reaches the FPU with
no dequant op in the graph. Converting an existing int4 GPTQ checkpoint into
them is not viable (measured ~9.8 % normalized error for 4-bit, because GPTQ is
uniform int4 with a zero point and block-float is symmetric with one shared
exponent per 16 values).

What *is* viable is quantizing onto that grid in the first place: GPTQ's solve is
grid-agnostic -- it needs a quantizer ``Q`` and its error, then minimises
Hessian-weighted error column by column. This module supplies ``Q`` for the
Tenstorrent grid so the quantizer can target a natively-supported format and
produce a checkpoint that needs no dequant at inference.

Grid definition (from ``tech_reports/data_formats``): values are grouped into
blocks of 16 that share one exponent, taken from the largest magnitude in the
block. Each value keeps a sign and the remaining bits of mantissa, with one
mantissa bit implicit:

===========  =====  ===========  ===================
format       bits   stored mant  representable levels
===========  =====  ===========  ===================
``bfloat8_b``   8        6        ~127 per exponent
``bfloat4_b``   4        2        7 per exponent
===========  =====  ===========  ===================

Rounding is round-to-nearest-even, matching the hardware behaviour described in
that report.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import torch

from .calibration import iter_calibration_windows


#: Bits per element for each supported block-float format.
BLOCKFLOAT_BITS = {"bfloat8_b": 8, "bfloat4_b": 4}
#: Values per shared exponent.
BLOCKFLOAT_BLOCK = 16


def stored_mantissa_bits(total_bits: int) -> int:
    """Return the explicitly stored mantissa bits for a block-float width."""

    # One sign bit, one implicit mantissa bit, the rest are stored mantissa.
    return max(total_bits - 2, 1)


def quantize_blockfloat(
    weight: torch.Tensor, total_bits: int, *, block: int = BLOCKFLOAT_BLOCK
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Round ``weight`` onto a block-float grid.

    Returns ``(dequantized, shared_exponent)``. The last dimension is split into
    blocks of ``block`` values that share one exponent, which is how the hardware
    stores it. The result is what the FPU would see after the unpack stage
    expands the stored format.
    """

    if weight.dim() < 1:
        raise ValueError("weight must have at least one dimension")
    if block <= 0:
        raise ValueError(f"block must be positive, got {block}")
    if total_bits not in (4, 8):
        raise ValueError(f"unsupported block-float width {total_bits}; expected 4 or 8")

    x = weight.to(torch.float32)
    shape = x.shape
    last = shape[-1]
    pad = (-last) % block
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
    blocks = x.reshape(*shape[:-1], x.shape[-1] // block, block)

    magnitude = blocks.abs().amax(dim=-1, keepdim=True)
    # Shared exponent from the block maximum; floor(log2) keeps the largest
    # value inside the mantissa range rather than scaling it out.
    safe = magnitude.clamp_min(torch.finfo(torch.float32).tiny)
    base_exponent = torch.floor(torch.log2(safe))

    mant_bits = stored_mantissa_bits(total_bits)
    # ``mant_bits`` explicit bits plus one implicit leading bit.
    levels = float(1 << mant_bits)
    # Clamp to the representable range for an exponent: [0, 2) in normalised
    # space with one implicit bit, i.e. below 2x the shared scale.
    limit = 2.0 - 1.0 / levels

    def _round_with(exp: torch.Tensor) -> torch.Tensor:
        scale = torch.pow(2.0, exp)
        rounded = torch.round(blocks / scale * levels) / levels
        return rounded.clamp(min=-limit, max=limit) * scale

    # ``floor(log2(block_max))`` is not always the exponent with the lowest
    # error: the grid only reaches ``limit * 2**exp``, so a block whose maximum
    # sits in the top quarter of a binade clips by up to ``2 - limit`` of its
    # largest value. Neighbouring exponents trade that clipping against a
    # coarser step, and the winner differs per block. Pick it per block by
    # measured squared error; the grid itself is unchanged.
    best = _round_with(base_exponent)
    best_err = (best - blocks).pow(2).sum(dim=-1, keepdim=True)
    best_exp = base_exponent
    for delta in (-1, 1):
        cand = _round_with(base_exponent + delta)
        err = (cand - blocks).pow(2).sum(dim=-1, keepdim=True)
        take = err < best_err
        best = torch.where(take.expand_as(best), cand, best)
        best_exp = torch.where(take, base_exponent + delta, best_exp)
        best_err = torch.where(take, err, best_err)

    exponent = best_exp
    out = best.reshape(*shape[:-1], x.shape[-1])
    if pad:
        out = out[..., :last]
    return out.contiguous(), exponent.squeeze(-1)


def quantize_bfloat8_b(weight: torch.Tensor) -> torch.Tensor:
    """Round ``weight`` the way ``bfloat8_b`` storage would round it."""

    return quantize_blockfloat(weight, 8)[0]


def quantize_bfloat4_b(weight: torch.Tensor) -> torch.Tensor:
    """Round ``weight`` the way ``bfloat4_b`` storage would round it."""

    return quantize_blockfloat(weight, 4)[0]


def quantize_uniform_int4(weight: torch.Tensor, *, group_size: int = 128) -> torch.Tensor:
    """Reference: symmetric uniform int4 with a per-group scale (GPTQ's grid)."""

    x = weight.to(torch.float32)
    last = x.shape[-1]
    groups = (last + group_size - 1) // group_size
    pad = groups * group_size - last
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
    blocks = x.reshape(*x.shape[:-1], groups, group_size)
    scale = blocks.abs().amax(dim=-1, keepdim=True) / 7.0
    scale = scale.clamp_min(torch.finfo(torch.float32).tiny)
    q = torch.round(blocks / scale).clamp(-8, 7)
    out = (q * scale).reshape(*x.shape[:-1], x.shape[-1])
    return (out[..., :last] if pad else out).contiguous()


def relative_error(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    """Normalized L1 error, the metric used elsewhere in this repository."""

    ref = reference.to(torch.float32)
    return float((candidate.to(torch.float32) - ref).abs().sum() / ref.abs().sum().clamp_min(1e-6))


def _entropy_bits(weight: torch.Tensor, *, group_size: int = 128) -> float:
    """Approximate bits/element a shared-exponent format adds for its exponents."""

    last = weight.shape[-1]
    return 8.0 / group_size if last % BLOCKFLOAT_BLOCK == 0 else 0.0


__all__ = [
    "BLOCKFLOAT_BITS",
    "BLOCKFLOAT_BLOCK",
    "quantize_bfloat4_b",
    "quantize_bfloat8_b",
    "quantize_blockfloat",
    "quantize_uniform_int4",
    "relative_error",
    "stored_mantissa_bits",
]


def _block_exponents(weight: torch.Tensor, block: int = BLOCKFLOAT_BLOCK) -> torch.Tensor:
    """Shared exponent per 16-wide block along the storage axis.

    The hardware groups 16 consecutive values of a row, so the exponent must be
    fixed from that grouping -- not per output channel -- or the solve produces
    values the format cannot store.
    """

    w = weight.to(torch.float32)
    pad = (-w.shape[-1]) % block
    if pad:
        w = torch.nn.functional.pad(w, (0, pad))
    blocks = w.reshape(*w.shape[:-1], -1, block)
    safe = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
    return torch.floor(torch.log2(safe)).squeeze(-1)


def _round_with_exponents(
    x: torch.Tensor, exponents: torch.Tensor, bits: int, block: int = BLOCKFLOAT_BLOCK
) -> torch.Tensor:
    """Round ``x`` onto a *fixed* set of block exponents."""

    w = x.to(torch.float32)
    last = w.shape[-1]
    pad = (-last) % block
    if pad:
        w = torch.nn.functional.pad(w, (0, pad))
    blocks = w.reshape(*w.shape[:-1], -1, block)
    scale = torch.pow(2.0, exponents)
    levels = float(1 << stored_mantissa_bits(bits))
    q = torch.round(blocks / scale * levels) / levels
    q = q.clamp(min=-(2.0 - 1.0 / levels), max=(2.0 - 1.0 / levels))
    out = (q * scale).reshape(*w.shape[:-1], w.shape[-1])
    return (out[..., :last] if pad else out).contiguous()


def gptq_solve_blockfloat(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    *,
    bits: int = 4,
    damp: float = 0.01,
    block: int = BLOCKFLOAT_BLOCK,
    grid: str = "blockfloat",
) -> torch.Tensor:
    """GPTQ error-feedback solve whose quantizer is the block-float grid.

    GPTQ'"'s algorithm is grid-agnostic: it needs a quantizer `Q(w)` and its error,
    then solves column by column minimising Hessian-weighted error. Nothing in the
    derivation requires a uniform grid -- that is only what GPTQ implementations
    conventionally use because it packs cleanly into nibbles.

    Substituting a Tenstorrent block-float grid therefore produces weights that are
    **natively `bfloat4_b`/`bfloat8_b` as stored**, so inference needs no dequant
    step at all: the unpack stage expands the checkpoint in the data path.

    `hessian` is `[in_features, in_features]` (``X^T X`` over calibration data).
    Returns the quantized weights with the same shape as `weight`
    (`[out_features, in_features]`), using GPTQ'"'s usual ordering: the reduction
    dimension is the *last* axis and `hessian` indexes it.

    This is the honest form of "GPTQ for bf4". Re-encoding an int4 checkpoint onto
    this grid is not the same thing and measures far worse.
    """

    w = weight.detach().to(torch.float32).clone()
    if w.dim() != 2:
        raise ValueError(f"weight must be 2-D, got {tuple(w.shape)}")
    out_features, in_features = w.shape
    if tuple(hessian.shape) != (in_features, in_features):
        raise ValueError(
            f"hessian must be [{in_features}, {in_features}], got {tuple(hessian.shape)}"
        )
    if bits not in (4, 8):
        raise ValueError(f"bits must be 4 or 8, got {bits}")
    if grid not in ("blockfloat", "uniform"):
        raise ValueError(f"grid must be 'blockfloat' or 'uniform', got {grid!r}")
    use_blockfloat = grid == "blockfloat"

    h = hessian.detach().to(torch.float32).clone()
    # Damp as GPTQ does, so a rank-deficient Hessian still inverts.
    dead = torch.diag(h) == 0
    h[dead, dead] = 1.0
    w[:, dead] = 0.0
    mean = torch.mean(torch.diag(h))
    h += torch.eye(in_features, dtype=h.dtype) * (damp * mean)

    # GPTQ needs the full inverse here. Taking ``triu`` of a lower-triangular
    # Cholesky factor leaves only the diagonal, which silently disables error
    # feedback (measured: the solve then matched plain rounding exactly).
    try:
        hinv = torch.linalg.inv(h)
    except Exception:  # pragma: no cover - singular Hessian
        hinv = torch.linalg.pinv(h)

    # The hardware fixes one exponent per 16-wide block of a row, so derive those
    # exponents from the *original* weights before the solve starts. Recomputing
    # them per column would let them drift and would make the result un-storable.
    fixed_exponents = _block_exponents(w, block) if use_blockfloat else None

    q = torch.zeros_like(w)
    for i in range(in_features):
        col = w[:, i]
        if use_blockfloat:
            # Exponent is fixed per row-block from the current (error-feedback)
            # weights, and the element is snapped onto that grid.
            block_idx = i // block
            exp = fixed_exponents[:, block_idx:block_idx + 1]
            levels = float(1 << stored_mantissa_bits(bits))
            scale = torch.pow(2.0, exp)
            unit = torch.round(col.unsqueeze(-1) / scale * levels) / levels
            unit = unit.clamp(min=-(2.0 - 1.0 / levels), max=(2.0 - 1.0 / levels))
            quant = (unit * scale).squeeze(-1)
        else:
            # Symmetric uniform grid with a per-column scale: GPTQ's usual choice.
            levels = float((1 << (bits - 1)) - 1)
            scale = col.abs().max().clamp_min(torch.finfo(torch.float32).tiny) / levels
            quant = torch.round(col / scale).clamp(-levels - 1, levels) * scale
        q[:, i] = quant
        err = (col - quant) / hinv[i, i]
        if i + 1 < in_features:
            w[:, i + 1:] -= err.unsqueeze(-1) @ hinv[i, i + 1:].unsqueeze(0)
    return q


__all__ += [
    "BLOCKFLOAT_SAVE_FORMAT_VERSION",
    "decompose_blockfloat",
    "load_blockfloat_into_model",
    "recompose_blockfloat",
    "expand_native_layer",
    "load_native_checkpoint_into_dense_model",
    "read_blockfloat_checkpoint",
    "save_blockfloat_checkpoint",
    "collect_layer_hessians",
    "quantize_model_blockfloat",
]


# --- end-to-end: quantize a dense model onto the TT block-float grid ---------
#
# The grid quantizer above is per-layer. This is the pipeline that turns a dense
# checkpoint into one whose weights are natively bfloat4_b/bfloat8_b, by
# collecting per-layer Hessians over calibration text and running the GPTQ solve
# against the grid. The result needs no inference-time dequant: the unpack stage
# expands it in the data path.

# There is deliberately no default calibration text. A single synthetic passage
# still tunes GPTQ's error feedback to that passage even when the windows stride
# correctly: on OPT-125M, 409 tokens of one passage gave perplexity 296 against
# 18.3 dense, while real text at the same sample budget gave 20.3. Calibration
# data is model- and domain-specific, so the caller must supply it.


def _decoder_blocks(model) -> list:
    """Return the transformer blocks in order, for the common HF layouts."""

    for path in ("model.decoder.layers", "model.layers", "transformer.h",
                 "gpt_neox.layers", "model.encoder.layers"):
        node = model
        try:
            for part in path.split("."):
                node = getattr(node, part)
        except AttributeError:
            continue
        return list(node)
    raise ValueError(
        "could not locate decoder blocks; expected one of model.decoder.layers, "
        "model.layers, transformer.h, gpt_neox.layers, model.encoder.layers"
    )


def collect_layer_hessians(
    model,
    tokenizer,
    *,
    layers: Optional[list] = None,
    text: Union[str, Sequence[str]],
    nsamples: int = 32,
    seqlen: int = 256,
) -> dict:
    """Collect ``X^T X`` for every quantizable linear, by replaying calibration text.

    Returns ``{layer_name: hessian}`` with one float32 Hessian per layer, plus the
    total sample count under the key ``"__nsamples__"``. Hooks capture each
    linear's input activations in order, so the Hessian lines up with the weights
    the solve will quantize.
    """

    import torch.nn as nn

    blocks = layers if layers is not None else _decoder_blocks(model)
    targets: dict = {}
    for index, block in enumerate(blocks):
        for name, module in block.named_modules():
            if isinstance(module, nn.Linear):
                targets[f"{index}.{name}"] = module
    if not targets:
        raise ValueError("no nn.Linear modules found inside the decoder blocks")

    hessians = dict.fromkeys(targets)
    counts = dict.fromkeys(targets, 0)
    # q/k/v share one input, and so do gate/up, so ``X^T X`` is computed once per
    # distinct activation rather than once per linear. Keyed on the tensor's
    # address with the tensor kept alive, so a freed buffer cannot alias a later
    # layer's input; the cache is cleared for every window. The reuse is exact --
    # it is the same matrix for the same activation.
    window_cache: dict = {}

    def make_hook(name):
        def hook(_module, inputs, _output):
            x = inputs[0]
            key = (x.data_ptr(), tuple(x.shape))
            cached = window_cache.get(key)
            if cached is None:
                # float32 is required here, not a default worth tuning.
                # Accumulating this in a narrow dtype was measured and rejected:
                # bf16 makes the square GEMM 3.4x faster (0.09 s against 0.31 s
                # for one 8192-wide X^T X over 2048 tokens) and its Hessian is
                # only 0.27 % from float32, which looks harmless -- but on
                # OPT-125M the quantized model came out with NaN weights and NaN
                # perplexity. fp16 is worse on both counts: 25.9 s for the same
                # GEMM (84x slower, no usable CPU kernels) and it overflows on
                # activation outliers. Never use bf16 or fp16 for the Hessian.
                xf = x.detach().to(torch.float32)
                if xf.dim() > 2:
                    xf = xf.reshape(-1, xf.shape[-1])
                cached = (x, xf.T @ xf, int(xf.shape[0]))
                window_cache[key] = cached
            h, rows = cached[1], cached[2]
            hessians[name] = h if hessians[name] is None else hessians[name] + h
            counts[name] += rows
        return hook

    handles = [module.register_forward_hook(make_hook(name))
               for name, module in targets.items()]
    # Only decoder activations are needed, so run the backbone and skip the output
    # projection: the lm_head GEMM is about 20 % of a forward pass on a 1B model
    # and its logits are never read here. Models without a usable ``.model``
    # backbone fall back to the full forward. This does not change any activation
    # the hooks see.
    backbone = getattr(model, "model", None)
    runner = model
    if (
        isinstance(backbone, nn.Module)
        and type(backbone).forward is not nn.Module.forward  # not a bare container
    ):
        runner = backbone
    try:
        model.eval()
        runner.eval()
        with torch.no_grad():
            # Windowing lives in one place so a quantizer cannot reintroduce the
            # "same passage every sample" bug that made this Hessian useless.
            for window in iter_calibration_windows(
                tokenizer, text, seqlen=seqlen, nsamples=nsamples
            ):
                window_cache.clear()
                runner(window)
    finally:
        for handle in handles:
            handle.remove()

    out = {name: h for name, h in hessians.items() if h is not None}
    out["__nsamples__"] = nsamples
    return out


def quantize_model_blockfloat(
    model,
    tokenizer,
    *,
    bits=4,
    damp: float = 0.01,
    nsamples: int = 32,
    text: Union[str, Sequence[str]],
    verbose: bool = False,
) -> dict:
    """Quantize every decoder linear onto the TT block-float grid, in place.

    Returns a report of ``{layer_name: {"norm_l1": ..., "on_grid": bool}}``. The
    weights are replaced with grid-quantized values, so saving the model yields a
    checkpoint that loads as native ``bfloat4_b``/``bfloat8_b`` with no dequant.

    ``bits`` may be a single width or a pattern -> width mapping (see
    :func:`_layer_bits`). Widths are chosen per layer, so a checkpoint can keep
    ``bfloat4_b`` on the projections that tolerate it and spend ``bfloat8_b`` on
    the ones that carry the most output error.
    """

    from .logger import setup_logger

    log = setup_logger()
    hessians = collect_layer_hessians(
        model, tokenizer, nsamples=nsamples, text=text
    )
    hessians.pop("__nsamples__", None)

    blocks = _decoder_blocks(model)
    report = {}
    for index, block in enumerate(blocks):
        for name, module in block.named_modules():
            key = f"{index}.{name}"
            h = hessians.get(key)
            if h is None:
                continue
            layer_bits = _layer_bits(bits, key)
            # Clone: ``detach()`` shares storage with the parameter, so copy_()
            # below would mutate the reference and make the error metric read 0.
            weight = module.weight.detach().clone()
            try:
                quantized = gptq_solve_blockfloat(
                    weight, h, bits=layer_bits, damp=damp, grid="blockfloat"
                )
            except Exception as exc:  # pragma: no cover - per-layer guard
                log.warning(f"TTBlockFloat: skipping {key}: {exc}")
                continue
            with torch.no_grad():
                module.weight.copy_(quantized.to(module.weight.dtype))
            norm_l1 = relative_error(weight, quantized)
            on_grid = torch.allclose(
                quantized, quantize_blockfloat(quantized, layer_bits)[0], atol=1e-6
            )
            report[key] = {
                "norm_l1": norm_l1,
                "on_grid": bool(on_grid),
                "bits": layer_bits,
            }
            if verbose:
                log.info(f"TTBlockFloat: {key} norm_l1={norm_l1:.4f} on_grid={on_grid}")
    return report


# --- packed checkpoint format ----------------------------------------------
#
# A grid-quantized weight is fully described by one small integer per element plus
# one exponent per 16-value block, so it can be stored at bf4 density WITHOUT
# re-quantising at load time:
#
#   codes:      uint8 per element, value in [0, 2*mantissa_bits] -- 0..8 for bf4
#   exponents:  int8 per 16-value block
#   dequant:    value = code / levels * 2**exponent
#
# That is exact: it is the same arithmetic the grid quantiser used, so loading
# reproduces the quantised weights bit-for-bit. Because the codes are already on
# the hardware grid, the loader can either expand them to floats or feed them
# straight to a TT kernel, with no host-side re-quantisation.

BLOCKFLOAT_SAVE_FORMAT_VERSION = 1

#: Tensor payload file. safetensors is the format: no pickle, memory-mappable, and
#: it is what the rest of the repo already ships. ``blockfloat.pt`` is still read
#: so checkpoints written before the switch keep loading.
BLOCKFLOAT_PAYLOAD = "blockfloat.safetensors"
BLOCKFLOAT_PAYLOAD_LEGACY = "blockfloat.pt"


def decompose_blockfloat(
    weight: torch.Tensor, bits: int = 4, block: int = BLOCKFLOAT_BLOCK
) -> "tuple[torch.Tensor, torch.Tensor]":
    """Split a grid-quantized weight into integer codes and per-block exponents.

    Raises if ``weight`` is not exactly on the grid, so a checkpoint can never
    silently claim to be native when it is not.
    """

    w = weight.to(torch.float32)
    rows, cols = w.shape
    if cols % block:
        raise ValueError(f"in_features={cols} must be a multiple of {block}")
    levels = 1 << stored_mantissa_bits(bits)
    # Grid codes span [-(2*levels-1), +(2*levels-1)]: one implicit mantissa bit
    # lets a block hold values up to just under 2x its shared scale. That is
    # 2**(bits-1) levels per sign for 4-bit, so the code fits a nibble exactly.
    offset = 2 * levels - 1
    # Recover each block's exponent. floor(log2(max)) is right only when the block
    # max sits in the lower half of its binade, so search the neighbours and keep
    # the candidate that reproduces the weights exactly. ``-1`` is in the list
    # because :func:`quantize_blockfloat` picks the lowest-error exponent per
    # block, which can be one step below ``floor(log2(max))``.
    base = _block_exponents(w, block)
    for shift in (0, 1, -1, 2):
        candidate = (base + shift).to(torch.float32)
        scale = torch.pow(2.0, candidate).repeat_interleave(block, dim=1)
        codes = torch.round(w / scale * levels)
        if codes.abs().max() > offset:
            continue
        if torch.equal(codes / levels * scale, w):
            packed = (codes + offset).to(torch.uint8)
            return packed, candidate.to(torch.int8)
    raise ValueError("weight is not exactly on the block-float grid")


def recompose_blockfloat(
    codes: torch.Tensor, exponents: torch.Tensor, bits: int = 4, block: int = BLOCKFLOAT_BLOCK
) -> torch.Tensor:
    """Rebuild exact grid weights from codes plus exponents (inverse of decompose)."""

    levels = 1 << stored_mantissa_bits(bits)
    offset = 2 * levels - 1
    scale = torch.pow(2.0, exponents.to(torch.float32)).repeat_interleave(block, dim=1)
    return ((codes.to(torch.float32) - offset) / levels * scale).contiguous()


def _pack_codes(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack codes at their natural density: two per byte for 4-bit, one for 8-bit."""

    flat = codes.reshape(-1)
    if bits == 4:
        if flat.numel() % 2:
            flat = torch.nn.functional.pad(flat, (0, 1))
        lo = flat[0::2].to(torch.int16)
        hi = flat[1::2].to(torch.int16)
        return (lo | (hi << 4)).to(torch.uint8)
    return flat.to(torch.uint8)


def _unpack_codes(packed: torch.Tensor, bits: int, shape: "tuple") -> torch.Tensor:
    """Inverse of :func:`_pack_codes`."""

    flat = packed.reshape(-1)
    if bits == 4:
        lo = (flat & 0x0F).to(torch.int16)
        hi = ((flat >> 4) & 0x0F).to(torch.int16)
        out = torch.stack([lo, hi], dim=1).reshape(-1)
        return out[: shape[0] * shape[1]].to(torch.uint8).reshape(shape)
    return flat[: shape[0] * shape[1]].to(torch.uint8).reshape(shape)


def _default_bits(bits) -> int:
    """The width recorded as the checkpoint default when ``bits`` is a mapping."""

    if isinstance(bits, dict):
        return int(bits.get("*", 4))
    return int(bits)


def _layer_bits(bits, layer_key: str) -> int:
    """Resolve the storage width for one layer.

    ``bits`` is either a single width or a mapping of pattern to width. An exact
    key wins, then the longest matching substring pattern, then ``"*"``. Longest
    match matters because ``"self_attn"`` and ``"q_proj"`` can both be present
    and the more specific rule should win.
    """

    if not isinstance(bits, dict):
        return int(bits)
    if layer_key in bits:
        return int(bits[layer_key])
    matches = [pat for pat in bits if pat != "*" and pat in layer_key]
    if matches:
        return int(bits[max(matches, key=len)])
    return int(bits.get("*", 4))


def save_blockfloat_checkpoint(
    model,
    output_dir: str,
    *,
    bits=4,
    metadata: "Optional[dict]" = None,
) -> dict:
    """Write grid-quantized decoder linears as a bf4/bf8-native checkpoint.

    Weights become ``<layer>.blockfloat_codes`` (uint8) plus
    ``<layer>.blockfloat_exponents`` (int8), so the stored payload is 1 byte per
    element for bf4 -- the same order as int4 packing without a second quantisation
    step at load. Returns a summary of what was written.

    ``bits`` may be a single width, or a mapping of layer-name pattern to width
    with an optional ``"*"`` default. Mixed widths are what a real checkpoint
    wants: ``bfloat8_b`` costs one byte per weight and is close to lossless,
    while ``bfloat4_b`` is roughly a 2.8-bit format, so spending the extra byte
    on the layers that carry the most output error is far cheaper than spending
    it everywhere.
    """

    import json
    import os

    from .logger import setup_logger

    log = setup_logger()
    blocks = _decoder_blocks(model)
    tensors: dict = {}
    summary: dict = {}
    default_bits = _default_bits(bits)

    for index, block in enumerate(blocks):
        for name, module in block.named_modules():
            if not hasattr(module, "weight") or module.weight is None:
                continue
            if module.weight.dim() != 2:
                continue
            key = f"{index}.{name}"
            layer_bits = _layer_bits(bits, key)
            w = module.weight.detach().to(torch.float32)
            try:
                codes, exponents = decompose_blockfloat(w, bits=layer_bits)
            except ValueError:
                # Not quantized (or not on the grid): keep it as dense float.
                tensors[f"{key}.weight"] = w.to(torch.float16)
                continue
            tensors[f"{key}.blockfloat_codes"] = _pack_codes(codes, layer_bits)
            tensors[f"{key}.blockfloat_code_shape"] = torch.tensor(codes.shape)
            tensors[f"{key}.blockfloat_exponents"] = exponents
            if getattr(module, "bias", None) is not None:
                tensors[f"{key}.bias"] = module.bias.detach().float().to(torch.float16)
            packed = tensors[f"{key}.blockfloat_codes"]
            summary[key] = {
                "shape": list(w.shape),
                "bits": layer_bits,
                "codes_bytes": int(packed.numel()),
                "numel": int(codes.numel()),
                "exponent_bytes": int(exponents.numel()),
            }

    os.makedirs(output_dir, exist_ok=True)
    from safetensors.torch import save_file

    save_file(
        {key: value.contiguous() for key, value in tensors.items()},
        os.path.join(output_dir, BLOCKFLOAT_PAYLOAD),
        metadata={
            "format": "tt-blockfloat",
            "version": str(BLOCKFLOAT_SAVE_FORMAT_VERSION),
            "bits": str(default_bits),
            "block": str(BLOCKFLOAT_BLOCK),
        },
    )
    # Caller metadata goes first: the format fields below describe how the payload
    # is encoded and are what the loader dispatches on, so they must win. A caller
    # that passed ``bits`` or ``layers`` would otherwise silently corrupt the file.
    meta = dict(metadata or {})
    meta.update({
        "format": "tt-blockfloat",
        "version": BLOCKFLOAT_SAVE_FORMAT_VERSION,
        "bits": default_bits,
        "block": BLOCKFLOAT_BLOCK,
        "stored_mantissa_bits": stored_mantissa_bits(default_bits),
        "layers": summary,
    })
    widths = sorted({entry["bits"] for entry in summary.values()})
    if len(widths) > 1:
        meta["mixed_bits"] = True
        meta["bits_per_layer"] = {key: entry["bits"] for key, entry in summary.items()}
    with open(os.path.join(output_dir, "blockfloat.json"), "w") as handle:
        json.dump(meta, handle, indent=1)
    total = sum(v["codes_bytes"] + v["exponent_bytes"] for v in summary.values())
    log.info(
        f"TTBlockFloat: wrote {len(summary)} layers to {output_dir} "
        f"({total / 1024 / 1024:.2f} MiB payload, "
        f"{'/'.join(str(w) for w in widths)}-bit)"
    )
    return {
        "layers": len(summary),
        "payload_bytes": total,
        "bits": default_bits,
        "bits_per_layer": widths,
    }


def _load_blockfloat_payload(checkpoint_dir: str) -> dict:
    """Read the tensor payload, preferring safetensors over the legacy pickle."""

    import os

    safetensors_path = os.path.join(checkpoint_dir, BLOCKFLOAT_PAYLOAD)
    if os.path.isfile(safetensors_path):
        from safetensors.torch import load_file

        return load_file(safetensors_path)

    legacy_path = os.path.join(checkpoint_dir, BLOCKFLOAT_PAYLOAD_LEGACY)
    if os.path.isfile(legacy_path):
        return torch.load(legacy_path, map_location="cpu")

    raise FileNotFoundError(
        f"no {BLOCKFLOAT_PAYLOAD} (or legacy {BLOCKFLOAT_PAYLOAD_LEGACY}) in "
        f"{checkpoint_dir}"
    )


def load_blockfloat_into_model(model, checkpoint_dir: str) -> dict:
    """Restore native grid weights into ``model`` from a saved checkpoint.

    No quantisation happens here -- the codes are already on the hardware grid, so
    this is arithmetic-free reconstruction and lands on the identical values.
    """

    # One implementation: the dense-restore path already resolves per-layer bit
    # widths and validates the checkpoint header, so duplicating the unpack here
    # would only be a second place to forget a mixed-width checkpoint.
    return load_native_checkpoint_into_dense_model(model, checkpoint_dir)


# --- serving a native checkpoint -------------------------------------------
#
# The point of saving codes instead of floats is that loading is arithmetic-free.
# These helpers expose the saved format to a caller that wants to feed the device
# directly, and provide a dense fallback so a native checkpoint is still usable
# with ordinary PyTorch where a TT kernel is not appropriate.

def read_blockfloat_checkpoint(checkpoint_dir: str) -> dict:
    """Read a native checkpoint into ``{layer_name: (codes, exponents, bias)}``.

    Codes stay packed exactly as stored; use :func:`recompose_blockfloat` (or the
    un-packed helper) to expand one layer.
    """

    import json
    import os

    meta_path = os.path.join(checkpoint_dir, "blockfloat.json")
    with open(meta_path) as handle:
        meta = json.load(handle)
    if meta.get("format") != "tt-blockfloat":
        raise ValueError(f"{meta_path} is not a tt-blockfloat checkpoint")
    bits = int(meta["bits"])
    tensors = _load_blockfloat_payload(checkpoint_dir)

    layers = {}
    for key in meta.get("layers", {}):
        # A mixed-width checkpoint records the width per layer; older ones only
        # have the global value.
        layer_bits = int(meta["layers"][key].get("bits", bits))
        codes = tensors.get(f"{key}.blockfloat_codes")
        exponents = tensors.get(f"{key}.blockfloat_exponents")
        if codes is None or exponents is None:
            continue
        shape = tensors.get(f"{key}.blockfloat_code_shape")
        if shape is not None:
            codes = _unpack_codes(codes, layer_bits, tuple(int(v) for v in shape))
        layers[key] = {
            "codes": codes,
            "exponents": exponents,
            "bias": tensors.get(f"{key}.bias"),
            "shape": tuple(meta["layers"][key]["shape"]),
            "bits": layer_bits,
        }
    return {"bits": bits, "block": int(meta["block"]), "layers": layers, "meta": meta}


def expand_native_layer(entry: dict) -> torch.Tensor:
    """Expand one native layer's codes to float weights (exact, no rounding)."""

    return recompose_blockfloat(
        entry["codes"], entry["exponents"], bits=entry["bits"]
    ).reshape(entry["shape"])


def load_native_checkpoint_into_dense_model(model, checkpoint_dir: str) -> dict:
    """Populate a dense model from a native checkpoint.

    This is the bridge for paths that cannot consume codes directly: it reconstructs
    the exact grid weights with no quantisation, so the result matches
    :func:`load_blockfloat_into_model` bit-for-bit while allowing callers to build
    the model however they like (for example on `meta` to skip a float allocation).
    """

    payload = read_blockfloat_checkpoint(checkpoint_dir)
    tensors = payload["layers"]
    blocks = _decoder_blocks(model)
    restored = {}
    with torch.no_grad():
        for index, block in enumerate(blocks):
            for name, module in block.named_modules():
                key = f"{index}.{name}"
                entry = tensors.get(key)
                if entry is None or getattr(module, "weight", None) is None:
                    continue
                weights = recompose_blockfloat(
                    entry["codes"], entry["exponents"], bits=entry["bits"]
                ).reshape(entry["shape"])
                module.weight.copy_(weights.to(module.weight.dtype))
                if entry["bias"] is not None and getattr(module, "bias", None) is not None:
                    module.bias.copy_(entry["bias"].to(module.bias.dtype))
                restored[key] = list(entry["shape"])
    return restored
# --- standard Hugging Face checkpoint layout -------------------------------
#
# A published checkpoint is an ordinary HF model directory: ``model.safetensors``
# (or ``model-0000N-of-0000M.safetensors`` shards) plus
# ``model.safetensors.index.json``. There is no per-role payload file and no
# sidecar manifest -- the index is the manifest, exactly as it is for every other
# HF checkpoint, and the loader reads it the same way.
#
# Tensor naming follows the module that owns it, which is what
# ``load_state_dict`` already understands: a quantized linear stores
# ``<prefix>.codes`` / ``.exponents`` / ``.code_shape`` (its buffer names) plus
# ``.bias``, while everything dense stores a normal ``.weight``.

INDEX_FILE = "model.safetensors.index.json"
SINGLE_SHARD_FILE = "model.safetensors"
SHARD_PATTERN = "model-{index:05d}-of-{total:05d}.safetensors"

#: Shard size limit. HF's own default is 5 GiB, but a 1B-model checkpoint is well
#: under 1 GiB, so the smaller default keeps the artifact in more than one shard
#: and exercises the index on every save.
DEFAULT_MAX_SHARD_BYTES = 500 * 1024**2


def _write_shards(tensors: dict, output_dir: str, max_shard_bytes: int) -> dict:
    """Write HF-style shards plus the standard index. Returns the weight map."""

    import json
    import os

    from safetensors.torch import save_file

    names = sorted(tensors)
    sizes = {name: tensors[name].numel() * tensors[name].element_size() for name in names}
    total_bytes = sum(sizes.values())

    shards, current, current_bytes = [], {}, 0
    for name in names:
        if current and current_bytes + sizes[name] > max_shard_bytes:
            shards.append(current)
            current, current_bytes = {}, 0
        current[name] = tensors[name]
        current_bytes += sizes[name]
    if current:
        shards.append(current)

    total = len(shards)
    weight_map = {}
    for index, shard in enumerate(shards, start=1):
        filename = (
            SINGLE_SHARD_FILE
            if total == 1
            else SHARD_PATTERN.format(index=index, total=total)
        )
        save_file(
            {key: value.contiguous() for key, value in shard.items()},
            os.path.join(output_dir, filename),
            metadata={"format": "pt"},
        )
        for key in shard:
            weight_map[key] = filename

    with open(os.path.join(output_dir, INDEX_FILE), "w") as handle:
        json.dump(
            {"metadata": {"total_size": total_bytes}, "weight_map": weight_map},
            handle,
            indent=2,
        )
    return {"weight_map": weight_map, "shards": total, "total_bytes": total_bytes}


def read_hf_weight_map(checkpoint_dir: str) -> dict:
    """Read ``model.safetensors.index.json``, or synthesise one for a single file."""

    import json
    import os

    index_path = os.path.join(checkpoint_dir, INDEX_FILE)
    if os.path.isfile(index_path):
        with open(index_path) as handle:
            weight_map = json.load(handle).get("weight_map") or {}
        if weight_map:
            return {str(key): str(value) for key, value in weight_map.items()}

    single = os.path.join(checkpoint_dir, SINGLE_SHARD_FILE)
    if os.path.isfile(single):
        from safetensors import safe_open

        with safe_open(single, framework="pt") as handle:
            return dict.fromkeys(handle.keys(), SINGLE_SHARD_FILE)

    raise FileNotFoundError(
        f"{checkpoint_dir} has no {INDEX_FILE} and no {SINGLE_SHARD_FILE}"
    )


def load_hf_state_dict(checkpoint_dir: str) -> dict:
    """Load every tensor named by the checkpoint's index."""

    import os

    from safetensors.torch import load_file

    weight_map = read_hf_weight_map(checkpoint_dir)
    by_shard: dict = {}
    for name, shard in weight_map.items():
        by_shard.setdefault(shard, []).append(name)

    state = {}
    for shard, names in by_shard.items():
        tensors = load_file(os.path.join(checkpoint_dir, shard))
        for name in names:
            if name in tensors:
                state[name] = tensors[name]
    return state

def _quantized_prefixes(model) -> list:
    """Module paths this checkpoint stores as block-float codes.

    Decoder linears plus the input embedding. The embedding is included because
    a vocabulary table is the single largest tensor in a small model: as codes at
    ``bfloat8_b`` it is 279 MiB against 501 MiB of dense float16.
    """

    import torch.nn as nn

    prefixes = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding) and name.endswith("embed_tokens"):
            prefixes.append(name)
        elif isinstance(module, nn.Linear) and ".layers." in name:
            prefixes.append(name)
    return prefixes


#: Modeling file shipped with the checkpoint so plain ``transformers`` can load
#: it. It only swaps in the block-float modules; the storage format is the
#: ordinary shards-plus-index layout.
MODELING_FILE = "modeling_tt_blockfloat.py"

_MODELING_SOURCE = """\
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
\"\"\"Modeling entry point for a Tenstorrent block-float checkpoint.

The weights are stored in the standard shards-plus-index layout; the only thing
that differs from a dense checkpoint is that the quantized modules keep packed
mantissa codes instead of a dense ``weight``. This swaps those modules in so a
plain ``AutoModelForCausalLM.from_pretrained`` produces a working model rather
than one with randomly initialised decoder layers.

Requires ``gptqmodel``.
\"\"\"

from transformers import LlamaForCausalLM

from gptqmodel.nn_modules.qlinear.tt_blockfloat import swap_blockfloat_modules


class TTBlockfloatForCausalLM(LlamaForCausalLM):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        quant = dict(getattr(config, "quantization_config", None) or {})
        swap_blockfloat_modules(
            self,
            bits_per_layer=dict(quant.get("bits_per_layer") or {}),
            default_bits=int(quant.get("bits", 4)),
            embed_bits=int(quant.get("embed_bits", 8)),
            block=int(quant.get("block", 16)),
        )

    def tie_weights(self, *args, **kwargs):
        # The tied head shares the embedding's codes by reference, so there is
        # no parameter to copy and nothing to re-tie after loading.
        return None

    def mark_tied_weights_as_initialized(self, loading_info):
        # ``transformers`` walks the tied-weight map after loading and looks up
        # each key with ``get_parameter``. The head has no dense parameter to
        # find, because it reads the embedding's codes, so there is nothing to
        # mark and the default implementation would raise.
        return None
"""


def save_blockfloat_model(
    model,
    output_dir: str,
    *,
    bits=4,
    embed_bits: int = 8,
    tokenizer=None,
    metadata: "Optional[dict]" = None,
    max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
) -> dict:
    """Write a self-contained checkpoint in the standard HF layout.

    The directory is an ordinary HF model: ``model.safetensors`` shards plus
    ``model.safetensors.index.json``. There is no per-role payload file and no
    sidecar manifest; the index is the manifest.

    Quantized tensors are named after the buffers their module owns
    (``<prefix>.codes`` / ``.exponents`` / ``.code_shape``), so ``load_state_dict``
    needs no translation table. The embedding is stored the same way, at
    ``embed_bits``.
    """

    import json
    import os

    os.makedirs(output_dir, exist_ok=True)
    tensors: dict = {}
    bits_per_layer: dict = {}
    default_bits = _default_bits(bits)
    tied = bool(getattr(model.config, "tie_word_embeddings", False))

    for prefix in _quantized_prefixes(model):
        module = model.get_submodule(prefix)
        is_embedding = not hasattr(module, "in_features")
        layer_width = embed_bits if is_embedding else _layer_bits(bits, prefix)
        weight = module.weight.detach().to(torch.float32)
        try:
            codes, exponents = decompose_blockfloat(weight, bits=layer_width)
        except ValueError:
            # Not on the grid (or not quantized): keep it dense like any other
            # float tensor rather than claiming a format it does not have.
            tensors[f"{prefix}.weight"] = weight.to(torch.float16).contiguous()
            continue
        tensors[f"{prefix}.codes"] = _pack_codes(codes, layer_width).contiguous()
        tensors[f"{prefix}.code_shape"] = torch.tensor(
            list(codes.shape), dtype=torch.int64
        )
        tensors[f"{prefix}.exponents"] = exponents.to(torch.int8).contiguous()
        bits_per_layer[prefix] = layer_width
        if not is_embedding and getattr(module, "bias", None) is not None:
            tensors[f"{prefix}.bias"] = module.bias.detach().to(torch.float16).contiguous()

    # Everything else is dense: norms, and the head only when it is not tied.
    quantized = set(bits_per_layer)
    for name, param in model.named_parameters():
        if name.endswith("embed_tokens.weight"):
            continue
        if name.endswith("lm_head.weight") and tied:
            continue
        if name in tensors:
            continue
        # A quantized module's parameters are already represented by its codes;
        # writing its dense ``.weight`` too would ship both forms and make the
        # loader reject the checkpoint for having nowhere to put the float one.
        if name.rsplit(".", 1)[0] in quantized:
            continue
        tensors[name] = param.detach().to(torch.float16).contiguous()

    written = _write_shards(tensors, output_dir, max_shard_bytes)

    config = model.config.to_dict()
    config["torch_dtype"] = "float16"
    quantization_config = {
        "quant_method": "tt-blockfloat",
        "checkpoint_format": "tt-blockfloat",
        "bits": default_bits,
        "block": BLOCKFLOAT_BLOCK,
        "group_size": BLOCKFLOAT_BLOCK,
        "stored_mantissa_bits": stored_mantissa_bits(default_bits),
        "sym": True,
        "desc_act": False,
        "lm_head": True,
        "lm_head_quant_bits": embed_bits if tied else 0,
        "lm_head_tied_to_embedding": tied,
        "embed_bits": embed_bits,
        "bits_per_layer": bits_per_layer,
        "meta": dict(metadata or {}),
    }
    widths = sorted(set(bits_per_layer.values()))
    if len(widths) > 1:
        quantization_config["mixed_bits"] = True
    config["quantization_config"] = quantization_config

    # Ship the modeling file so plain ``transformers`` can load this directory.
    # Without it, ``from_pretrained`` reports the codes as unexpected, leaves the
    # decoder layers at their random initial values, and returns a model that
    # generates garbage with no error.
    config["auto_map"] = {"AutoModelForCausalLM": f"{MODELING_FILE[:-3]}.TTBlockfloatForCausalLM"}
    with open(os.path.join(output_dir, "config.json"), "w") as handle:
        json.dump(config, handle, indent=2)
    with open(os.path.join(output_dir, MODELING_FILE), "w") as handle:
        handle.write(_MODELING_SOURCE)

    if tokenizer is not None:
        tokenizer.save_pretrained(output_dir)

    return {
        "layers": len(bits_per_layer),
        "payload_bytes": written["total_bytes"],
        "shards": written["shards"],
        "bits": default_bits,
        "bits_per_layer": widths,
        "embed_bits": embed_bits,
        "tied": tied,
        "dense_tensors": len(tensors) - 3 * len(bits_per_layer),
    }


def _resolve_layer_widths(state: dict, quantization_config: dict) -> dict:
    """Per-module storage width, preferring the payload over the config.

    A wrong width in ``config.json`` would decode every code at the wrong scale
    and produce plausible-looking garbage, so the packed length is treated as
    authoritative when the two disagree.
    """

    from ..nn_modules.qlinear.tt_blockfloat import packed_code_bytes

    declared = {
        str(key): int(value)
        for key, value in (quantization_config.get("bits_per_layer") or {}).items()
    }
    default_bits = int(quantization_config.get("bits", 4))

    widths = {}
    for name in state:
        if not name.endswith(".codes"):
            continue
        prefix = name[: -len(".codes")]
        shape = tuple(int(v) for v in state.get(f"{prefix}.code_shape", []))
        if len(shape) != 2:
            continue
        width = declared.get(prefix, default_bits)
        numel = shape[0] * shape[1]
        if state[name].numel() != packed_code_bytes(numel, width):
            width = 8 if state[name].numel() == numel else 4
        widths[prefix] = width
    return widths


def build_blockfloat_model(
    checkpoint_dir: str, *, dtype: torch.dtype = torch.float32
) -> "tuple":
    """Build the torch model for a checkpoint directory.

    Returns ``(model, restored_layers, info)``. Shared by the library loader and
    the ``GPTQModel.load`` route so both read the checkpoint identically.
    """

    import transformers
    from transformers import AutoConfig
    from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

    from ..nn_modules.qlinear.tt_blockfloat import swap_blockfloat_modules

    # Read the index first: a directory with no standard HF weight files is not a
    # checkpoint at all, and saying so is more useful than a config parse error.
    state = load_hf_state_dict(checkpoint_dir)
    if not state:
        raise ValueError(f"{checkpoint_dir} has no tensors in its index")

    config = AutoConfig.from_pretrained(checkpoint_dir)
    quantization_config = getattr(config, "quantization_config", None) or {}
    if not isinstance(quantization_config, dict):
        quantization_config = dict(quantization_config)

    widths = _resolve_layer_widths(state, quantization_config)
    if not widths:
        raise ValueError(
            f"{checkpoint_dir} has no block-float tensors; it is a dense checkpoint"
        )

    # Build through the concrete class rather than ``AutoModelForCausalLM`` so a
    # checkpoint's own ``auto_map`` cannot make this path import remote code.
    class_name = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.get(config.model_type)
    if class_name is None:
        raise ValueError(
            f"{checkpoint_dir} has unsupported model_type `{config.model_type}`"
        )
    model = getattr(transformers, class_name)._from_config(config).to(dtype=dtype)

    restored = swap_blockfloat_modules(
        model,
        bits_per_layer=widths,
        default_bits=int(quantization_config.get("bits", 4)),
        embed_bits=int(quantization_config.get("embed_bits", 8)),
        block=int(quantization_config.get("block", BLOCKFLOAT_BLOCK)),
        quantized_prefixes=sorted(widths),
    )

    missing, unexpected = model.load_state_dict(state, strict=False)
    tied = bool(getattr(config, "tie_word_embeddings", False))
    unexpected = [
        name for name in unexpected if not (name.endswith("lm_head.weight") and tied)
    ]
    if unexpected:
        raise ValueError(
            f"{checkpoint_dir} contains tensors this config has no home for: "
            f"{sorted(unexpected)[:5]}"
        )
    required = [
        name for name in missing if not (name.endswith("lm_head.weight") and tied)
    ]
    if required:
        raise ValueError(
            f"{checkpoint_dir} does not supply {len(required)} required tensors, "
            f"e.g. {sorted(required)[:5]}"
        )

    model.eval()
    return model, restored, {
        "bits": int(quantization_config.get("bits", 4)),
        "quantization_config": quantization_config,
    }


def load_blockfloat_model(
    checkpoint_dir: str, *, dtype: torch.dtype = torch.float32
) -> "tuple":
    """Load a checkpoint written by :func:`save_blockfloat_model`.

    Reads the standard HF index, so nothing here depends on this project's file
    naming: any tensor named ``<prefix>.codes`` swaps that module for the
    block-float kernel, and everything else loads as an ordinary parameter.
    """

    model, restored, _info = build_blockfloat_model(checkpoint_dir, dtype=dtype)
    return model, restored


__all__ += [
    "INDEX_FILE",
    "MODELING_FILE",
    "SINGLE_SHARD_FILE",
    "build_blockfloat_model",
    "load_blockfloat_model",
    "load_hf_state_dict",
    "read_hf_weight_map",
    "save_blockfloat_model",
]
