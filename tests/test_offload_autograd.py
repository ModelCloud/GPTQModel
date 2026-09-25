# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from accelerate import disk_offload

from gptqmodel.utils.offload import undo_offload_to_disk


class _TiedModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=True)
        self.embedding = torch.nn.Embedding(8, 4, dtype=torch.float64)
        self.output = torch.nn.Linear(4, 8, bias=False, dtype=torch.float64)
        self.register_buffer("scale", torch.ones(4, dtype=torch.float64))
        self.register_buffer("indices", torch.arange(4))
        self.tie_weights()

    def tie_weights(self) -> None:
        self.output.weight = self.embedding.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(x * self.scale)


class _CopyTiedModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=True)
        self.embedding = torch.nn.Embedding(8, 4)
        self.output = torch.nn.Linear(4, 8, bias=False)
        with torch.no_grad():
            self.embedding.weight.fill_(0.5)
            self.output.weight.zero_()

    def tie_weights(self) -> None:
        self.output.weight.copy_(self.embedding.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(x)


class _RestoreTargetModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.eye(2))
        self.register_buffer("scale", torch.ones(2))
        self.child = torch.nn.Linear(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self.scale) @ self.weight.T + self.child(x)


@pytest.mark.parametrize("outer_inference", [False, True])
@pytest.mark.parametrize("offload_buffers", [False, True])
def test_disk_restore_produces_autograd_safe_tensors(
    tmp_path: Path,
    outer_inference: bool,
    offload_buffers: bool,
) -> None:
    model = _TiedModel()
    expected = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    disk_offload(
        model,
        offload_dir=str(tmp_path / "offload"),
        execution_device=torch.device("cpu"),
        offload_buffers=offload_buffers,
    )
    with torch.inference_mode(outer_inference):
        assert undo_offload_to_disk(model) is model
    for name, actual in model.state_dict().items():
        torch.testing.assert_close(actual, expected[name])
        assert not actual.is_inference()
        assert actual._version >= 0
    assert model.embedding.weight is model.output.weight
    assert not any(hasattr(sub, "_hf_hook") for sub in model.modules())
    inputs = torch.ones(2, 4, dtype=torch.float64, requires_grad=True)
    model(inputs).sum().backward()
    assert inputs.grad is not None
    assert model.output.weight.grad is not None
    assert model.output.weight.requires_grad


@pytest.mark.parametrize("outer_inference", [False, True])
def test_restore_allows_tie_weights_to_copy_into_trainable_parameter(
    tmp_path: Path, outer_inference: bool
) -> None:
    model = _CopyTiedModel()
    disk_offload(
        model,
        offload_dir=str(tmp_path / "offload"),
        execution_device=torch.device("cpu"),
    )

    with torch.inference_mode(outer_inference):
        undo_offload_to_disk(model)

    torch.testing.assert_close(model.output.weight, model.embedding.weight)
    assert not model.output.weight.is_inference()
    inputs = torch.ones(2, 4, requires_grad=True)
    model(inputs).sum().backward()
    assert model.output.weight.grad is not None
    assert inputs.grad is not None


@pytest.mark.parametrize("offload_buffers", [False, True])
def test_disk_restore_preserves_requested_dtype_after_hook_detach(
    tmp_path: Path, offload_buffers: bool
) -> None:
    model = _RestoreTargetModel()
    disk_offload(
        model,
        offload_dir=str(tmp_path / "offload"),
        execution_device=torch.device("cpu"),
        offload_buffers=offload_buffers,
    )

    with torch.inference_mode():
        undo_offload_to_disk(model, dtype=torch.float64)

    assert all(param.dtype == torch.float64 for param in model.parameters())
    assert model.scale.dtype == (torch.float64 if offload_buffers else torch.float32)
    assert not any(hasattr(sub, "_hf_hook") for sub in model.modules())
    inputs = torch.ones(2, 2, dtype=torch.float64, requires_grad=True)
    model(inputs).sum().backward()
    assert inputs.grad is not None
    assert model.weight.grad is not None


def test_disk_restore_leaves_unrelated_meta_tensors_alone(tmp_path: Path) -> None:
    model = torch.nn.Module()
    model.offloaded = torch.nn.Linear(2, 2)
    model.uninitialized = torch.nn.Linear(2, 2, device="meta")
    disk_offload(
        model.offloaded,
        offload_dir=str(tmp_path / "offload"),
        execution_device=torch.device("cpu"),
    )

    undo_offload_to_disk(model, dtype=torch.float64)

    assert model.offloaded.weight.dtype == torch.float64
    assert model.uninitialized.weight.is_meta


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_disk_restore_preserves_requested_gpu_after_hook_detach(tmp_path: Path) -> None:
    model = _RestoreTargetModel()
    disk_offload(
        model,
        offload_dir=str(tmp_path / "offload"),
        execution_device=torch.device("cuda:0"),
        offload_buffers=True,
    )

    with torch.inference_mode():
        undo_offload_to_disk(model, device=torch.device("cuda:0"), dtype=torch.float64)

    assert all(param.device == torch.device("cuda:0") and param.dtype == torch.float64 for param in model.parameters())
    assert model.scale.device == torch.device("cuda:0")
    assert model.scale.dtype == torch.float64
    assert not any(hasattr(sub, "_hf_hook") for sub in model.modules())
    inputs = torch.ones(2, 2, device="cuda:0", dtype=torch.float64, requires_grad=True)
    model(inputs).sum().backward()
    assert inputs.grad is not None
    assert model.weight.grad is not None
