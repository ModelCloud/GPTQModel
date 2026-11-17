# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""CPU format arithmetic for solver feasibility experiments, not a checkpoint API.

Only finite weights are accepted. Arithmetic is FP64; stored scales are rounded
*before* solving. Payloads use canonical low-nibble/low-bit-first row packing,
not a vendor GEMM swizzle or a GPTQ/AWQ checkpoint layout.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FormatSpec:
    name: str
    element: str
    block_size: int
    scale: str

    @property
    def bits(self):
        return 4 if self.element in ("e2m1", "int4") else 8


FORMATS = (
    FormatSpec("int4", "int4", 32, "fp32"),
    FormatSpec("mxfp4", "e2m1", 32, "e8m0"),
    FormatSpec("nvfp4", "e2m1", 16, "e4m3"),
    FormatSpec("mxfp8_e4m3", "e4m3", 32, "e8m0"),
    FormatSpec("mxfp8_e5m2", "e5m2", 32, "e8m0"),
    FormatSpec("fp8_e4m3", "e4m3", 0, "fp32"),
)


def format_spec(name):
    for spec in FORMATS:
        if spec.name == name:
            return spec
    raise ValueError(f"Unknown reference format {name!r}; NVFP8 has no assumed encoding")


def positive_grid(element):
    """Finite nonnegative codebook in unsigned code order, including zero."""
    if element == "int4":
        return torch.arange(16, dtype=torch.float64)
    if element == "e2m1":
        return torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float64)
    if element not in ("e4m3", "e5m2"):
        raise ValueError(f"Unknown element {element!r}")
    mantissa, bias, last = (3, 7, 126) if element == "e4m3" else (2, 15, 123)
    code = torch.arange(last + 1, dtype=torch.int64)
    exponent, fraction = code >> mantissa, (code % (1 << mantissa)).double() / (1 << mantissa)
    return torch.where(exponent == 0, fraction * 2. ** (1 - bias),
                       (1 + fraction) * torch.exp2((exponent - bias).double()))


def encode_elements(values, element):
    """Saturating round-to-nearest-even with signed zero, without native FP4/8."""
    if values.device.type != "cpu" or not values.is_floating_point() or not torch.isfinite(values).all():
        raise ValueError("Element encoding requires finite CPU floating values")
    if element == "int4":
        return values.round().clamp(0, 15).to(torch.uint8)
    grid = positive_grid(element)
    magnitude = values.double().abs().contiguous()
    hi = torch.searchsorted(grid, magnitude).clamp(max=len(grid) - 1)
    lo = (hi - 1).clamp(min=0)
    # Compare to the midpoint, not two subtractions: threshold neighbours must
    # not collapse into artificial ties through cancellation.
    midpoint = (grid[lo] + grid[hi]) / 2
    upper = (magnitude > midpoint) | ((magnitude == midpoint) & (hi % 2 == 0))
    code = torch.where(upper, hi, lo)
    sign = 8 if element == "e2m1" else 128
    return (code | (torch.signbit(values).long() * sign)).to(torch.uint8)


def decode_elements(codes, element):
    if codes.device.type != "cpu" or codes.dtype != torch.uint8:
        raise ValueError("Element decoding requires CPU uint8 codes")
    grid = positive_grid(element)
    if element == "int4":
        if (codes > 15).any():
            raise ValueError("Invalid INT4 code")
        return codes.double()
    sign = 8 if element == "e2m1" else 128
    index = codes.long() & (sign - 1)
    if (index >= len(grid)).any() or (element == "e2m1" and (codes > 15).any()):
        raise ValueError("Nonfinite or invalid element code")
    return torch.where((codes.long() & sign) != 0, -grid[index], grid[index])


def _matrix(value):
    if (value.device.type != "cpu" or value.ndim != 2 or min(value.shape) == 0
            or not value.is_floating_point() or not torch.isfinite(value).all()):
        raise ValueError("Reference arithmetic requires a finite nonempty CPU floating matrix")
    return value.detach().double().clone()


@dataclass(frozen=True)
class FormatGrid:
    spec: FormatSpec
    shape: tuple[int, int]
    scales: torch.Tensor
    tensor_scale: torch.Tensor | None = None
    zeros: torch.Tensor | None = None

    def __post_init__(self):
        rows, columns = self.shape
        block = self.spec.block_size
        if rows <= 0 or columns <= 0 or (block and columns % block):
            raise ValueError("Invalid format grid shape")
        shape = (rows, columns // block) if block else (1, 1)
        dtype = torch.float32 if self.spec.scale == "fp32" else torch.uint8
        if self.scales.device.type != "cpu" or self.scales.shape != shape or self.scales.dtype != dtype:
            raise ValueError("Scale storage does not match the format")
        if self.spec.scale == "e8m0" and (self.scales == 255).any():
            raise ValueError("E8M0 code 255 is NaN")
        if self.spec.scale == "e4m3":
            if (self.tensor_scale is None or self.tensor_scale.device.type != "cpu"
                    or self.tensor_scale.dtype != torch.float32 or self.tensor_scale.ndim != 0
                    or not torch.isfinite(self.tensor_scale) or self.tensor_scale <= 0
                    or (self.scales == 0).any() or (self.scales > 126).any()):
                raise ValueError("NVFP4 requires positive E4M3 and FP32 scales")
        elif self.tensor_scale is not None:
            raise ValueError("Only NVFP4 uses a second scale in this reference")
        if self.spec.scale == "fp32" and (not torch.isfinite(self.scales).all() or (self.scales <= 0).any()):
            raise ValueError("FP32 scales must be finite and positive")
        if self.spec.element == "int4":
            if (self.zeros is None or self.zeros.device.type != "cpu" or self.zeros.shape != shape
                    or self.zeros.dtype != torch.uint8 or (self.zeros > 15).any()):
                raise ValueError("INT4 requires uint8 zero points in [0,15]")
        elif self.zeros is not None:
            raise ValueError("Floating formats have no affine zero point")

    def expanded(self):
        if self.spec.scale == "e8m0":
            scale = torch.exp2(self.scales.double() - 127)
        elif self.spec.scale == "e4m3":
            scale = decode_elements(self.scales, "e4m3") * self.tensor_scale.double()
        else:
            scale = self.scales.double()
        if self.spec.block_size == 0:
            return scale.expand(self.shape)
        return scale.repeat_interleave(self.spec.block_size, dim=1)

    def zero_grid(self):
        if self.zeros is None:
            return torch.zeros(self.shape, dtype=torch.float64)
        return self.zeros.double().repeat_interleave(self.spec.block_size, dim=1)

    def encode(self, weight):
        weight = _matrix(weight)
        if tuple(weight.shape) != self.shape:
            raise ValueError("Weight shape differs from the fitted grid")
        normalized = weight / self.expanded()
        if self.zeros is not None:
            normalized = normalized + self.zero_grid()
        return encode_elements(normalized, self.spec.element)

    def decode(self, codes):
        return (decode_elements(codes, self.spec.element) - self.zero_grid()) * self.expanded()

    def pack(self, codes):
        if tuple(codes.shape) != self.shape:
            raise ValueError("Code shape differs from the fitted grid")
        decode_elements(codes, self.spec.element)
        if self.spec.element == "int4":
            lanes = codes.long().reshape(self.shape[0], -1, 8)
            payload = (lanes << (4 * torch.arange(8))).sum(-1).to(torch.int32)
        elif self.spec.bits == 4:
            payload = codes[:, ::2] | (codes[:, 1::2] << 4)
        else:
            payload = codes.clone()
        return FormatTensor(self, payload)


@dataclass(frozen=True)
class FormatTensor:
    grid: FormatGrid
    payload: torch.Tensor

    def codes(self):
        if self.grid.spec.element == "int4":
            return ((self.payload.long().unsqueeze(-1) >> (4 * torch.arange(8))) & 15).to(
                torch.uint8).reshape(self.grid.shape)
        if self.grid.spec.bits == 4:
            return torch.stack((self.payload & 15, self.payload >> 4), dim=-1).reshape(self.grid.shape)
        return self.payload.clone()

    def dequantize(self):
        return self.grid.decode(self.codes())

    @property
    def storage_bits(self):
        tensors = (self.payload, self.grid.scales, self.grid.tensor_scale, self.grid.zeros)
        return sum(t.numel() * t.element_size() * 8 for t in tensors if t is not None)


def fit_grid(weight, name):
    """Fit one legal grid; scale selection is a recipe, not the format itself.

    MX uses ceil(log2(amax / max_element)); NVFP4 uses RNE E4M3 local
    scales and an FP32 global decode multiplier. Tiny scales are floored to
    the minimum positive stored scale. INT4 is affine with FP32 scales.
    """
    weight = _matrix(weight)
    spec = format_spec(name)
    if spec.block_size and weight.shape[1] % spec.block_size:
        raise ValueError(f"{name} requires columns divisible by {spec.block_size}")
    shape = tuple(weight.shape)
    blocks = weight.reshape(shape[0], -1, spec.block_size) if spec.block_size else weight.reshape(1, 1, -1)
    amax = blocks.abs().amax(-1)
    tensor_scale, zeros = None, None
    if spec.element == "int4":
        low, high = blocks.amin(-1).clamp(max=0), blocks.amax(-1).clamp(min=0)
        scales = ((high - low) / 15).clamp(min=torch.finfo(torch.float32).tiny).float()
        scales = torch.where(high == low, torch.ones_like(scales), scales)
        zeros = (-low / scales.double()).round().clamp(0, 15).to(torch.uint8)
    elif spec.scale == "e8m0":
        maximum = positive_grid(spec.element)[-1]
        exponent = torch.ceil(torch.log2(amax / maximum)).clamp(-127, 127)
        scales = torch.where(amax == 0, 127, exponent + 127).to(torch.uint8)
    elif spec.scale == "e4m3":
        tensor_scale = (amax.max() / (6 * 448)).clamp(min=torch.finfo(torch.float32).tiny).float()
        local = amax / (6 * tensor_scale.double())
        scales = encode_elements(local.clamp(max=448), "e4m3").clamp(min=1)
    else:
        scales = (amax / positive_grid(spec.element)[-1]).clamp(min=torch.finfo(torch.float32).tiny).float()
        scales = torch.where(amax == 0, torch.ones_like(scales), scales)
    grid = FormatGrid(spec, shape, scales, tensor_scale, zeros)
    if not torch.isfinite(grid.expanded()).all() or (grid.expanded() <= 0).any():
        raise ValueError("Scale is not representable by the target format")
    return grid


def quantize_rtn(weight, name):
    grid = fit_grid(weight, name)
    return grid.pack(grid.encode(weight))
