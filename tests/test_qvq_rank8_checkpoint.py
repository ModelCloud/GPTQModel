import json

import pytest
import torch
from safetensors.torch import save_file

from gptqmodel.nn_modules.qlinear.qvq import QVQLinear
from gptqmodel.quantization.qvq_rank8_checkpoint import prepare_rank8_checkpoint_buffers


def shell():
    model = torch.nn.Module()
    model.proj = QVQLinear(bits=2, in_features=32, out_features=16, v2b2_p32=True, bank_count=2)
    return model


def payload():
    return {"proj.rank8_A": torch.zeros(32, 8, dtype=torch.float16),
            "proj.rank8_B": torch.zeros(8, 16, dtype=torch.float16),
            "proj.rank8_metadata": torch.zeros(12, dtype=torch.uint8)}


def test_rank8_sharded_header_allocation(tmp_path):
    tensors = payload()
    save_file({"proj.rank8_A": tensors["proj.rank8_A"]}, tmp_path / "a.safetensors")
    save_file({k: v for k, v in tensors.items() if k != "proj.rank8_A"}, tmp_path / "b.safetensors")
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {
        k: "a.safetensors" if k.endswith("rank8_A") else "b.safetensors" for k in tensors
    }}))
    model = shell().to("meta")
    prepare_rank8_checkpoint_buffers(model, index, set(tensors))
    for key, expected in tensors.items():
        actual = getattr(model.proj, key.split(".")[-1])
        assert actual.is_meta and actual.shape == expected.shape and actual.dtype == expected.dtype


@pytest.mark.parametrize("invalid", ["partial", "dtype", "shape"])
def test_rank8_invalid_header_does_not_mutate_shell(tmp_path, invalid):
    tensors = payload()
    if invalid == "partial":
        del tensors["proj.rank8_B"]
    elif invalid == "dtype":
        tensors["proj.rank8_B"] = tensors["proj.rank8_B"].float()
    else:
        tensors["proj.rank8_B"] = torch.zeros(16, 8, dtype=torch.float16)
    path = tmp_path / "model.safetensors"
    save_file(tensors, path)
    model = shell()
    with pytest.raises(ValueError, match="rank8"):
        prepare_rank8_checkpoint_buffers(model, path, set(tensors))
    assert model.proj.rank8_A is model.proj.rank8_B is model.proj.rank8_metadata is None


def test_window_only_loader_does_not_read_optional_payload():
    model = shell()
    prepare_rank8_checkpoint_buffers(model, "/nonexistent", {"proj.trellis"})
    assert model.proj.rank8_A is None
