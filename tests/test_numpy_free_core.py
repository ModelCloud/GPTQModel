# SPDX-License-Identifier: Apache-2.0
"""Guards the NumPy-free core: no direct imports, and graceful degradation.

GPT-QModel is moving toward installs where NumPy is not guaranteed. The modules
listed here must not import NumPy at all, and the checkpoint RNG capture must
tolerate its absence while remaining byte-identical when it is present.
"""

import ast
import builtins
import math
import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parent.parent

NUMPY_FREE_MODULES = (
    "gptqmodel/utils/tensor.py",
    "gptqmodel/utils/hf.py",
    "gptqmodel/utils/audio.py",
    "gptqmodel/utils/marlin.py",
    "gptqmodel/nn_modules/qlinear/__init__.py",
    "gptqmodel/nn_modules/qlinear/qqq.py",
    "gptqmodel/nn_modules/qlinear/marlin.py",
    "gptqmodel/quantization/rco.py",
    "gptqmodel/looper/gptq_checkpoint.py",
    "gptqmodel/looper/gsq_training_model.py",
    "gptqmodel/models/loader.py",
    "gptqmodel/utils/model.py",
    "gptqmodel/utils/offload.py",
    "gptqmodel/utils/exl3_projection_checkpoint.py",
    "gptqmodel/utils/calibration.py",
)


def _numpy_import_nodes(tree: ast.AST) -> list[ast.AST]:
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.split(".")[0] == "numpy" for alias in node.names):
                found.append(node)
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] == "numpy":
                found.append(node)
    return found


def _catches_module_not_found(handler: ast.ExceptHandler) -> bool:
    target = handler.type
    if target is None:
        return True  # bare except: still degrades instead of failing import
    names = target.elts if isinstance(target, ast.Tuple) else [target]
    return any(isinstance(item, ast.Name) and item.id == "ModuleNotFoundError" for item in names)


@pytest.mark.parametrize("relative_path", NUMPY_FREE_MODULES)
def test_module_scope_has_no_numpy_import(relative_path: str) -> None:
    path = REPO_ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"))
    offenders = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            offenders.extend(alias.name for alias in node.names if alias.name.split(".")[0] == "numpy")
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] == "numpy":
                offenders.append(node.module)
    assert not offenders, f"{relative_path} still imports numpy at module scope: {offenders}"


@pytest.mark.parametrize("relative_path", NUMPY_FREE_MODULES)
def test_function_local_numpy_imports_are_guarded(relative_path: str) -> None:
    """A lazy NumPy probe is only allowed behind ``except ModuleNotFoundError``."""
    tree = ast.parse((REPO_ROOT / relative_path).read_text(encoding="utf-8"))
    guarded = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Try) and any(_catches_module_not_found(h) for h in node.handlers):
            for child in ast.walk(node):
                guarded.add(id(child))
    unguarded = [node for node in _numpy_import_nodes(tree) if id(node) not in guarded and node not in tree.body]
    assert not unguarded, (
        f"{relative_path} imports numpy without a ModuleNotFoundError guard at "
        f"line(s) {[node.lineno for node in unguarded]}"
    )


def test_checkpoint_rng_capture_round_trips_when_numpy_is_present() -> None:
    pytest.importorskip("numpy")
    from gptqmodel.looper.gptq_checkpoint import _capture_numpy_rng, _restore_numpy_rng

    saved = _capture_numpy_rng()
    assert saved is not None
    assert _restore_numpy_rng(saved) is True


def test_checkpoint_rng_capture_degrades_without_numpy(monkeypatch) -> None:
    from gptqmodel.looper.gptq_checkpoint import _capture_numpy_rng, _restore_numpy_rng

    # The capture only consults sys.modules; it never imports NumPy.
    monkeypatch.delitem(sys.modules, "numpy", raising=False)
    assert _capture_numpy_rng() is None
    # A checkpoint captured with NumPy cannot be verified without it.
    assert _restore_numpy_rng(("MT19937", [1, 2, 3, 624], 0, 0, 0.0)) is False


def test_tensor_bytes_matches_numpy_tobytes() -> None:
    pytest.importorskip("numpy")
    from gptqmodel.utils.tensor import tensor_bytes

    tensor = torch.arange(-32, 32, dtype=torch.int16)
    assert tensor_bytes(tensor) == tensor.numpy().tobytes()
    assert tensor_bytes(torch.tensor([], dtype=torch.uint8)) == b""
    assert tensor_bytes(torch.arange(16)[3:9]) == torch.arange(16)[3:9].numpy().tobytes()
    assert tensor_bytes(torch.randn(4, 4, dtype=torch.bfloat16)).__len__() == 32


def test_tensor_bytes_does_not_import_numpy(monkeypatch) -> None:
    from gptqmodel.utils.tensor import tensor_bytes

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "numpy" or name.startswith("numpy."):
            raise ModuleNotFoundError("No module named 'numpy'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    tensor = torch.arange(8, dtype=torch.uint8)
    assert tensor_bytes(tensor) == bytes(range(8))


def _reference_dp(gained: torch.Tensor, steps: list[list[int]], budget: int) -> torch.Tensor:
    """Pure-Python DP used as an independent oracle (no NumPy, no Torch ops)."""
    n_layers, n_options = gained.shape
    neg_inf = float("-inf")
    table = [neg_inf] * (budget + 1)
    table[0] = 0.0
    choice = [[0] * (budget + 1) for _ in range(n_layers)]
    for layer in range(n_layers):
        updated = [neg_inf] * (budget + 1)
        for option in range(n_options):
            step = steps[layer][option]
            if step > budget:
                continue
            gain = float(gained[layer, option])
            for j in range(step, budget + 1):
                best = table[j - step] + gain
                if best > updated[j]:
                    updated[j] = best
                    choice[layer][j] = option
        table = updated
    if not any(math.isfinite(value) for value in table):
        raise RuntimeError("RCO rounding found no feasible assignment under the budget")
    position = max(range(len(table)), key=table.__getitem__)
    out = [0] * n_layers
    for layer in range(n_layers - 1, -1, -1):
        option = choice[layer][position]
        out[layer] = option
        position -= steps[layer][option]
    return torch.tensor(out, dtype=torch.long)


def test_rco_dp_paths_match_reference() -> None:
    """The compiled kernel and the Torch fallback must match an independent DP."""
    from gptqmodel.quantization import rco

    compiled = rco._rco_dp_compiled()
    for case in range(200):
        generator = torch.Generator().manual_seed(500 + case)
        n_layers = int(torch.randint(1, 7, (1,), generator=generator))
        n_options = int(torch.randint(1, 6, (1,), generator=generator))
        budget = int(torch.randint(1, 400, (1,), generator=generator))
        scores = torch.randn(n_layers, n_options, generator=generator, dtype=torch.float64)
        if case % 4 == 0:
            scores = scores.round()  # ties
        if case % 7 == 0:
            scores = scores * 1e3  # saturated
        unit_costs = torch.randint(1, max(2, budget), (n_layers, n_options), generator=generator)
        steps = unit_costs.tolist()

        try:
            expected = _reference_dp(scores, steps, budget)
        except RuntimeError as exc:
            # Both paths must agree that the budget is infeasible.
            with pytest.raises(RuntimeError, match=str(exc)):
                rco._budget_dp_torch(scores, steps, budget, n_layers, n_options)
            if compiled is not None:
                with pytest.raises(RuntimeError, match=str(exc)):
                    compiled(scores, unit_costs, budget)
            continue
        assert torch.equal(
            rco._budget_dp_torch(scores, steps, budget, n_layers, n_options), expected
        ), f"torch case {case}"
        if compiled is not None:
            assert torch.equal(compiled(scores, unit_costs, budget), expected), f"compiled case {case}"

    # A non-contiguous input must still hit the kernel's raw pointers correctly.
    noncontiguous = torch.randn(7, 4, dtype=torch.float64).t()
    costs = torch.randint(1, 40, (4, 7))
    steps = costs.tolist()
    expected = _reference_dp(noncontiguous, steps, 120)
    assert torch.equal(rco._budget_dp(noncontiguous, costs, 120, 4, 7), expected)
