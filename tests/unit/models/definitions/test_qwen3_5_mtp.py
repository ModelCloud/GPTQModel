import pytest
import torch
from safetensors.torch import save_file

from gptqmodel.models.definitions.qwen3_5 import Qwen3_5QModel
from gptqmodel.models.definitions.qwen3_5_text import Qwen3_5TextQModel
from gptqmodel.models.writer import (
    _merge_prefix_tensors_into_state_dict,
    _normalize_out_of_model_tensors_entries,
)


@pytest.mark.parametrize("model_cls", [Qwen3_5QModel, Qwen3_5TextQModel])
def test_qwen3_5_merges_mtp_tensors_from_auxiliary_safetensors(tmp_path, model_cls):
    mtp_tensor = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    non_mtp_tensor = torch.ones((2, 2), dtype=torch.float32)

    save_file(
        {
            "mtp.fc.weight": mtp_tensor,
            "model.layers.0.mlp.down_proj.weight": non_mtp_tensor,
        },
        str(tmp_path / "model-auxiliary.safetensors"),
    )

    state_dict = {}

    copy_files, prefix_entries = _normalize_out_of_model_tensors_entries(
        model_cls.out_of_model_tensors
    )

    assert copy_files == []
    assert prefix_entries == ["mtp"]

    _merge_prefix_tensors_into_state_dict(prefix_entries, str(tmp_path), state_dict)

    assert "mtp.fc.weight" in state_dict
    assert "model.layers.0.mlp.down_proj.weight" not in state_dict
    assert torch.equal(state_dict["mtp.fc.weight"].source, mtp_tensor)
