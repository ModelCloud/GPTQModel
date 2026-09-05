"""Experimental CPU approximations for candidates 15/16/27/28/30.

Inputs are caller-supplied reconstructed weights/codebooks; no checkpoint I/O,
calibration, backend registration or GPU execution occurs here. All fitting and
reconstruction uses float64. These post-hoc fits are not jointly trained P32
quantizers and establish neither model quality nor inference speed.

Storage reports describe a proposed byte-aligned payload, not a deployed format.
They include shape/config headers, padded tile codes, factors/scales and indices.
Python/container overhead and runtime scratch are excluded; tensor payload bytes
are reported separately. Caller must add retained P32 streams, banks, transforms,
shared LUTs and model metadata when computing complete model BPW.
"""

from dataclasses import dataclass
from math import prod

import torch

EXPERIMENTAL_APPROXIMATION = True


def _integer(value: int, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _input(x: torch.Tensor) -> torch.Tensor:
    if (
        x.device.type != "cpu"
        or not x.is_floating_point()
        or x.numel() == 0
        or x.ndim == 0
    ):
        raise ValueError("Expected a nonempty nonscalar floating CPU tensor")
    if not torch.isfinite(x).all():
        raise ValueError("Non-finite input")
    return x.detach().to(torch.float64).clone()


def _storage(parts: dict[str, int], tensors: tuple, weights: int) -> dict:
    _integer(weights, "logical_weights")
    total = sum(parts.values())
    return {
        "classification": "experimental_approximation",
        "accounting": "proposed_payload_not_serialized_or_model_total",
        "components_bytes": parts,
        "total_bytes": total,
        "logical_weights": weights,
        "effective_bpw": 8 * total / weights,
        "tensor_payload_bytes": sum(t.numel() * t.element_size() for t in tensors),
    }


@dataclass(frozen=True)
class CodebookFactorization:
    """Table [I,J,D]: additive C0[i,d]+C1[j,d] or sum_r A[d,i,r]B[d,r,j]."""

    left: torch.Tensor
    right: torch.Tensor
    additive: bool = False

    def decode(self) -> torch.Tensor:
        if self.additive:
            return self.left[:, None, :] + self.right[None, :, :]
        return (self.left @ self.right).permute(1, 2, 0).contiguous()

    def lookup(self, i: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
        """Equal-shape CPU int64 indices; no modulo repair of invalid codes."""
        ni, nj = (
            (self.left.shape[0], self.right.shape[0])
            if self.additive
            else (self.left.shape[1], self.right.shape[2])
        )
        for index, bound in ((i, ni), (j, nj)):
            if index.device.type != "cpu" or index.dtype != torch.int64:
                raise ValueError("Indices must be CPU int64")
            if bool(((index < 0) | (index >= bound)).any()):
                raise ValueError("Codebook index out of range")
        if i.shape != j.shape:
            raise ValueError("Index shapes must match")
        if self.additive:
            return self.left[i] + self.right[j]
        return (
            (self.left[:, i, :] * self.right[:, :, j].movedim(1, -1))
            .sum(-1)
            .movedim(0, -1)
        )

    def storage(self, logical_weights: int) -> dict:
        """Dictionary only; caller must inventory every encoded index/trellis stream."""
        return _storage(
            {
                "factors": (self.left.numel() + self.right.numel()) * 8,
                "shape_mode_rank_header": 6 * 8,
            },
            (self.left, self.right),
            logical_weights,
        )


def fit_additive_codebook(table: torch.Tensor) -> CodebookFactorization:
    """Candidate 15: unweighted least-squares additive projection, fixed index grid."""
    q = _input(table)
    if q.ndim != 3:
        raise ValueError("Expected [I,J,D] codebook")
    return CodebookFactorization(q.mean(1), q.mean(0) - q.mean((0, 1)), True)


def fit_tensor_product_codebook(
    table: torch.Tensor, rank: int = 1
) -> CodebookFactorization:
    """Candidate 27: independent truncated SVD per output coordinate, ranks 1/2/4.

    This is a sum of products on a fixed two-index grid, not whole-weight SVD.
    Index assignment and codebook learning under calibration remain external.
    """
    q = _input(table)
    if q.ndim != 3 or rank not in (1, 2, 4) or isinstance(rank, bool):
        raise ValueError("Expected [I,J,D] and rank 1, 2, or 4")
    _integer(rank, "rank")
    if rank > min(q.shape[:2]):
        raise ValueError("Rank exceeds codebook grid dimensions")
    u, s, vh = torch.linalg.svd(q.permute(2, 0, 1), full_matrices=False)
    return CodebookFactorization(
        (u[..., :rank] * s[..., None, :rank]).contiguous(), vh[:, :rank, :].contiguous()
    )


def _tiles(x: torch.Tensor, tile_size: int) -> torch.Tensor:
    _integer(tile_size, "tile_size")
    # Tile independently along the last dimension; never cross an output row.
    count = (x.shape[-1] + tile_size - 1) // tile_size
    return torch.nn.functional.pad(x, (0, count * tile_size - x.shape[-1])).reshape(
        -1, tile_size
    )


def _untile(tiles: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    return tiles.reshape(*shape[:-1], -1)[..., : shape[-1]].contiguous()


@dataclass(frozen=True)
class TileBasis:
    """Candidates 16/28: per-tile sum_r scales[t,r] * codes[t,r,k].

    Signed codes are {-1,+1} (proposed 0/1 encoding); ternary codes are
    {-1,0,+1} (proposed 0/1/2 encoding, 3 reserved). Codes remain int8 here.
    """

    codes: torch.Tensor
    scales: torch.Tensor
    shape: tuple[int, ...]
    ternary: bool

    def decode(self) -> torch.Tensor:
        return _untile(
            (self.codes.double() * self.scales[..., None]).sum(1), self.shape
        )

    def storage(self) -> dict:
        bits = 2 if self.ternary else 1
        tiles, rank, width = self.codes.shape
        return _storage(
            {
                "codes_including_padded_tails": tiles
                * rank
                * ((width * bits + 7) // 8),
                "scales": self.scales.numel() * 8,
                "shape_config_header": (len(self.shape) + 4) * 8,
            },
            (self.codes, self.scales),
            prod(self.shape),
        )


def fit_tile_basis(
    weights: torch.Tensor, rank: int = 1, tile_size: int = 64, ternary: bool = False
) -> TileBasis:
    """Greedy residual projection, no activation weighting or joint retraining.

    Ternary support minimizes each step's squared error over sorted-magnitude
    prefixes. Signed scale is mean absolute residual. Tail padding is excluded
    from fitting and included in storage. Earlier rank prefixes remain fixed.
    """
    x = _input(weights)
    _integer(rank, "rank")
    if not isinstance(ternary, bool):
        raise TypeError("ternary must be bool")
    residual = _tiles(x, tile_size)
    valid = _tiles(torch.ones_like(x), tile_size).bool()
    codes, scales = [], []
    for _ in range(rank):
        sign = torch.where(residual >= 0, 1.0, -1.0)
        if ternary:
            order = residual.abs().argsort(dim=-1, descending=True, stable=True)
            magnitude = residual.abs().gather(1, order)
            count = torch.arange(1, tile_size + 1, dtype=torch.float64)
            gain = magnitude.cumsum(-1).square() / count
            best = gain.argmax(-1) + 1
            support = torch.zeros_like(valid).scatter(
                1, order, count[None, :] <= best[:, None]
            )
            active = support & valid
        else:
            active = valid
        plane = sign * active
        scale = (residual * plane).sum(-1) / active.sum(-1).clamp_min(1)
        residual = residual - scale[:, None] * plane
        # Signed padding must still use a legal binary code; it is cropped on decode.
        stored = plane if ternary else torch.where(valid, plane, 1.0)
        codes.append(stored.to(torch.int8))
        scales.append(scale)
    return TileBasis(
        torch.stack(codes, 1), torch.stack(scales, 1), tuple(x.shape), ternary
    )


def orthonormal_walsh(x: torch.Tensor) -> torch.Tensor:
    """Sylvester-order orthonormal FWHT on last axis; self-inverse in real arithmetic."""
    y = _input(x)
    n = y.shape[-1]
    if n & (n - 1):
        raise ValueError("Walsh width must be a power of two")
    width = 1
    while width < n:
        blocks = y.reshape(*y.shape[:-1], -1, 2, width)
        a, b = blocks[..., 0, :], blocks[..., 1, :]
        y = torch.stack((a + b, a - b), -2).reshape(y.shape)
        width *= 2
    return y / n**0.5


@dataclass(frozen=True)
class SparseWalsh:
    """Candidate 30: row-local one-sided orthonormal Walsh with fixed top-k per tile."""

    indices: torch.Tensor
    values: torch.Tensor
    shape: tuple[int, ...]
    block_size: int

    def decode(self) -> torch.Tensor:
        coeff = torch.zeros(
            (self.indices.shape[0], self.block_size), dtype=torch.float64
        )
        coeff.scatter_(1, self.indices, self.values)
        return _untile(orthonormal_walsh(coeff), self.shape)

    def storage(self) -> dict:
        bits = (self.block_size - 1).bit_length()
        tiles, keep = self.indices.shape
        return _storage(
            {
                "indices": tiles * ((keep * bits + 7) // 8),
                "coefficients_including_tail_tiles": self.values.numel() * 8,
                "shape_config_header": (len(self.shape) + 4) * 8,
            },
            (self.indices, self.values),
            prod(self.shape),
        )


def encode_sparse_walsh(
    weights: torch.Tensor, block_size: int = 32, keep: int = 8
) -> SparseWalsh:
    """Keep largest-magnitude coefficients, ties resolved by ascending index.

    Zero-pad each row tail before transforming; crop only after inverse transform.
    Full keep is a floating-point round trip, not a bitwise exact P32 path.
    """
    x = _input(weights)
    _integer(block_size, "block_size")
    _integer(keep, "keep", 0)
    if block_size & (block_size - 1) or keep > block_size:
        raise ValueError("Require power-of-two block_size and 0 <= keep <= block_size")
    coeff = orthonormal_walsh(_tiles(x, block_size))
    indices = (
        coeff.abs().argsort(dim=-1, descending=True, stable=True)[:, :keep].contiguous()
    )
    return SparseWalsh(indices, coeff.gather(1, indices), tuple(x.shape), block_size)
