"""CPU reference algebra for experiments 19/20/29, not a native kernel deployment.

Convention: X[M,K], folded teacher W[K,N], correction D=A[K,r] B[r,N].
Capture teacher outputs from canonical FP32 P32 and native outputs from the actual
configured callable (including activation quantization, scales, bias and epilogue).
Use verified calibration text only; retain disjoint quality evaluation. Callbacks
must not mutate inputs or reuse mutable packed state across quantization calls.
The F6 snapshot is never opened or written here.

Integration must supply a deployed quantizer/callable, complete packed storage
bytes (codes, scales, zeros, padding, LUTs, metadata), and separately measure the
combined native + low-rank + sparse operator after casting/exporting factors.
These FP64 CPU fits are reference solutions, not latency or model-quality evidence.
Sparse application below is an unfused reference. Account for retained P32 blocks
separately, with disjoint support to avoid double-counting correction terms.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch

Tensor = torch.Tensor


def _matrix(t: Tensor) -> Tensor:
    if t.ndim != 2 or min(t.shape) == 0 or not t.is_floating_point() or not torch.isfinite(t).all():
        raise ValueError("expected a finite nonempty floating matrix")
    return t.detach().to(device="cpu", dtype=torch.float64)


def _count(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


@dataclass
class LowRank:
    a: Tensor
    b: Tensor
    activation_rank: int
    squared_error: float

    def dense(self) -> Tensor:
        return self.a @ self.b

    def __call__(self, x: Tensor) -> Tensor:
        """Reference arithmetic; deployment must explicitly cast/export factors."""
        return (_matrix(x) @ self.a) @ self.b


@torch.no_grad()
def fit_output_residual(x: Tensor, z: Tensor, rank: int, *, rcond: float | None = None) -> LowRank:
    """Minimize ||Z-XAB||_F subject to rank(AB)<=rank on retained singular space.

    X=U S V^T, C=U^T Z. Eckart-Young gives C_r; D=V S^-1 C_r.
    Thus loss = ||(I-UU^T)Z||^2 + sum_{j>r} sigma_j(C)^2.
    Unlike truncating pinv(X)Z, this minimizes *output* error. Nullspace
    coefficients are zero (minimum-norm representative). rcond truncation defines
    numerical activation rank; no ridge or normal equations are used.
    """
    _count(rank, "rank")
    x, z = _matrix(x), _matrix(z)
    if x.shape[0] != z.shape[0]:
        raise ValueError("X and Z row counts differ")
    if rcond is None:
        rcond = max(x.shape) * torch.finfo(x.dtype).eps
    if not 0 <= rcond < 1:
        raise ValueError("rcond must be finite and in [0,1)")
    u, s, vh = torch.linalg.svd(x, full_matrices=False)
    p = int((s > s[0] * rcond).sum())
    c = u[:, :p].T @ z
    left, values, right = torch.linalg.svd(c, full_matrices=False)
    r = min(rank, p, z.shape[1])
    a = (vh[:p].T / s[:p]) @ (left[:, :r] * values[:r])
    b = right[:r]
    error = float((z - (x @ a) @ b).square().sum())
    return LowRank(a, b, p, error)


@torch.no_grad()
def capture_residual(x: Tensor, teacher: Callable[[Tensor], Tensor], native: Callable[[Tensor], Tensor]) -> Tensor:
    """Execute both callbacks on original dtype/device; subtract on CPU in FP64."""
    _matrix(x)
    yt, yn = _matrix(teacher(x)), _matrix(native(x))
    if yt.shape != yn.shape or yt.shape[0] != x.shape[0]:
        raise ValueError("teacher/native output geometry mismatch")
    return yt - yn


@dataclass
class SparseResidual:
    """COO weight correction; indices are [nnz,2] int64 (input,output)."""
    indices: Tensor
    values: Tensor
    shape: tuple[int, int]

    def dense(self) -> Tensor:
        result = torch.zeros(self.shape, dtype=self.values.dtype)
        result[self.indices[:, 0], self.indices[:, 1]] = self.values
        return result

    def __call__(self, x: Tensor) -> Tensor:
        x = _matrix(x)
        result = torch.zeros((x.shape[0], self.shape[1]), dtype=self.values.dtype)
        result.index_add_(1, self.indices[:, 1], x[:, self.indices[:, 0]] * self.values)
        return result


@torch.no_grad()
def fit_sparse_residual(x: Tensor, z: Tensor, nnz: int) -> SparseResidual:
    """Greedy output-space pursuit, refitting each output's selected columns.

    Select maximum squared residual correlation / column energy. This is a
    budgeted heuristic, not the globally optimal sparse regression solution.
    Pass Z minus the actual low-rank output for experiment 29.
    """
    _count(nnz, "nnz")
    x, z = _matrix(x), _matrix(z)
    if x.shape[0] != z.shape[0] or nnz > x.shape[1] * z.shape[1]:
        raise ValueError("invalid sparse budget or row geometry")
    energy = x.square().sum(0)
    allowed = (energy > 0)[:, None].expand(x.shape[1], z.shape[1]).clone()
    weights = torch.zeros_like(allowed, dtype=x.dtype)
    residual = z.clone()
    selected = torch.zeros_like(allowed)
    for _ in range(nnz):
        scores = (x.T @ residual).square() / energy.clamp_min(torch.finfo(x.dtype).tiny)[:, None]
        scores[~allowed] = -1
        flat = int(scores.argmax())
        i, j = divmod(flat, z.shape[1])
        if scores[i, j] <= 0:
            break
        selected[i, j], allowed[i, j] = True, False
        columns = selected[:, j].nonzero().flatten()
        coef = torch.linalg.lstsq(x[:, columns], z[:, j], driver="gelsd").solution
        weights[columns, j] = coef
        residual[:, j] = z[:, j] - x[:, columns] @ coef
    indices = selected.nonzero()
    return SparseResidual(indices, weights[selected], tuple(weights.shape))


@dataclass
class NativeOperator:
    """Caller-owned immutable packed operator; bytes must include ALL native storage."""
    forward: Callable[[Tensor], Tensor]
    storage_bytes: Mapping[str, int]
    description: str


@dataclass
class RecoveryStep:
    native: NativeOperator
    low_rank: LowRank
    sparse: SparseResidual
    squared_error: float


@torch.no_grad()
def alternating_recovery(
    x: Tensor, teacher: Callable[[Tensor], Tensor], folded_weight: Tensor,
    quantize: Callable[[Tensor], NativeOperator], rank: int, *, iterations: int = 1, sparse_nnz: int = 0,
) -> list[RecoveryStep]:
    """19: iterations=1; 20: quantize(W-AB-S)/refit; 29: sparse_nnz>0.

    Each iteration fits fresh actual native output residuals, then sparse error
    remaining after low rank. Return every iterate: quantization is discrete and
    improvement is NOT guaranteed. Select on separate validation; never benchmarks.
    Quantize receives a fresh CPU FP64 KxN tensor; adapter handles device/layout.
    """
    _count(iterations, "iterations")
    _count(rank, "rank")
    _count(sparse_nnz, "sparse_nnz")
    xc, w = _matrix(x), _matrix(folded_weight)
    yt = _matrix(teacher(x))
    if xc.shape[1] != w.shape[0] or yt.shape != (xc.shape[0], w.shape[1]):
        raise ValueError("teacher, X and folded weight geometry mismatch")
    correction = torch.zeros_like(w)
    history = []
    for _ in range(iterations):
        native = quantize(w - correction)
        yn = _matrix(native.forward(x))
        if yn.shape != yt.shape:
            raise ValueError("native output geometry mismatch")
        z = yt - yn
        low = fit_output_residual(xc, z, rank)
        sparse = fit_sparse_residual(xc, z - low(xc), sparse_nnz)
        correction = low.dense() + sparse.dense()
        loss = float((z - low(xc) - sparse(xc)).square().sum())
        history.append(RecoveryStep(native, low, sparse, loss))
    return history


def storage_cost(step: RecoveryStep, *, extra_bytes: Mapping[str, int] | None = None) -> dict:
    """Actual tensor payload bytes, not assumed FP16: cast first, then recount.

    COO costs nnz*(2*index_bytes+value_bytes); low rank costs r*(K+N)*factor_bytes
    for equal factor dtypes. Add serialized headers, retained-block payloads,
    alignment and any deployment-specific copies via extra_bytes. Excludes fitting
    workspace; this is storage accounting, not peak runtime memory.
    """
    parts = {f"native/{k}": v for k, v in step.native.storage_bytes.items()}
    for name, tensor in (("A", step.low_rank.a), ("B", step.low_rank.b),
                         ("sparse_indices", step.sparse.indices), ("sparse_values", step.sparse.values)):
        parts[name] = tensor.numel() * tensor.element_size()
    if extra_bytes is not None:
        parts.update({f"extra/{k}": v for k, v in extra_bytes.items()})
    for value in parts.values():
        _count(value, "storage bytes")
    total = sum(parts.values())
    k, n = step.sparse.shape
    return {"components_bytes": parts, "total_bytes": total, "effective_bpw": 8 * total / (k * n)}
