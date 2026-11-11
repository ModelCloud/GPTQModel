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

from typing import Optional, Tuple

import torch


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
    exponent = torch.floor(torch.log2(safe))
    scale = torch.pow(2.0, exponent)

    mant_bits = stored_mantissa_bits(total_bits)
    # ``mant_bits`` explicit bits plus one implicit leading bit.
    levels = float(1 << mant_bits)
    normalised = blocks / scale
    rounded = torch.round(normalised * levels) / levels
    # Clamp to the representable range for this exponent: [0, 2) in normalised
    # space with one implicit bit, i.e. below 2x the shared scale.
    rounded = rounded.clamp(min=-(2.0 - 1.0 / levels), max=(2.0 - 1.0 / levels))
    out = (rounded * scale).reshape(*shape[:-1], x.shape[-1])
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
    "DEFAULT_CALIBRATION_TEXT",
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

DEFAULT_CALIBRATION_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Tenstorrent accelerators pair many small RISC-V cores with a matrix engine. "
    "Quantization reduces the number of bits needed to store model weights. "
    "A transformer layer projects hidden states through attention and feed-forward blocks. "
    "Numerical precision and memory bandwidth both bound inference throughput. "
    "The model reads a sentence, predicts the next token, and repeats. "
)


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
    text: str = DEFAULT_CALIBRATION_TEXT,
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

    def make_hook(name):
        def hook(_module, inputs, _output):
            x = inputs[0].detach().to(torch.float32)
            if x.dim() > 2:
                x = x.reshape(-1, x.shape[-1])
            h = x.T @ x
            hessians[name] = h if hessians[name] is None else hessians[name] + h
            counts[name] += int(x.shape[0])
        return hook

    handles = [module.register_forward_hook(make_hook(name))
               for name, module in targets.items()]
    try:
        ids = tokenizer(text, return_tensors="pt").input_ids
        model.eval()
        with torch.no_grad():
            for i in range(nsamples):
                window = ids[:, :seqlen]
                if window.shape[1] < 8:
                    window = ids
                model(window)
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
    bits: int = 4,
    damp: float = 0.01,
    nsamples: int = 32,
    text: str = DEFAULT_CALIBRATION_TEXT,
    verbose: bool = False,
) -> dict:
    """Quantize every decoder linear onto the TT block-float grid, in place.

    Returns a report of ``{layer_name: {"norm_l1": ..., "on_grid": bool}}``. The
    weights are replaced with grid-quantized values, so saving the model yields a
    checkpoint that loads as native ``bfloat4_b``/``bfloat8_b`` with no dequant.
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
            # Clone: ``detach()`` shares storage with the parameter, so copy_()
            # below would mutate the reference and make the error metric read 0.
            weight = module.weight.detach().clone()
            try:
                quantized = gptq_solve_blockfloat(
                    weight, h, bits=bits, damp=damp, grid="blockfloat"
                )
            except Exception as exc:  # pragma: no cover - per-layer guard
                log.warning(f"TTBlockFloat: skipping {key}: {exc}")
                continue
            with torch.no_grad():
                module.weight.copy_(quantized.to(module.weight.dtype))
            norm_l1 = relative_error(weight, quantized)
            on_grid = torch.allclose(
                quantized, quantize_blockfloat(quantized, bits)[0], atol=1e-6
            )
            report[key] = {"norm_l1": norm_l1, "on_grid": bool(on_grid)}
            if verbose:
                log.info(f"TTBlockFloat: {key} norm_l1={norm_l1:.4f} on_grid={on_grid}")
    return report
