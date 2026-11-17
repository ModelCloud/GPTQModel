# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Only GPTQ-layout methods may accept a model-wide `rotation` (#4666).

`rotation` is a base-class field, so every config accepted it. Only the
GPTQ-layout kernels apply the online Hadamard at inference, and the loader
refuses a rotated checkpoint unless its format is GPTQ/GPTQ_V2. Accepting the
field elsewhere fused a rotation nothing undoes: the in-session model was
silently wrong (+9 nll) and the saved checkpoint could not be loaded.

The backend gate in `quantize()` used to run *after* `fuse_layer_norms` and
`rotate_model`, so even a correct refusal left the caller's model rotated.
"""

import pytest
import torch
from test_resave_preserves_checkpoint_dtypes_4379 import CALIBRATION, _tiny_source

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.models import base as base_module
from gptqmodel.quantization.config import (
    AWQConfig,
    BitsAndBytesConfig,
    EXL3Config,
    FP8Config,
    GGUFConfig,
    GPTQConfig,
    MacaqConfig,
    NF12Config,
    ParoConfig,
    QQQConfig,
    RTNConfig,
    VPTQConfig,
)


pytestmark = [pytest.mark.cpu]


REJECTED = {
    "awq": lambda: AWQConfig(bits=4, group_size=128, rotation="hadamard"),
    "fp8": lambda: FP8Config(rotation="hadamard"),
    "bitsandbytes": lambda: BitsAndBytesConfig(rotation="hadamard"),
    "gguf": lambda: GGUFConfig(rotation="hadamard"),
    "qqq": lambda: QQQConfig(bits=4, group_size=128, rotation="hadamard"),
    "vptq": lambda: VPTQConfig(bits=4, group_size=128, rotation="hadamard"),
    "paroquant": lambda: ParoConfig(bits=4, group_size=128, rotation="hadamard"),
    "exl3": lambda: EXL3Config(bits=4, rotation="hadamard"),
    "nf12": lambda: NF12Config(rotation="hadamard"),
}


@pytest.mark.parametrize("build", list(REJECTED.values()), ids=list(REJECTED))
def test_non_gptq_layouts_reject_model_rotation(build):
    with pytest.raises(ValueError, match="requires a GPTQ or GPTQ_V2") as error:
        build()

    assert "No weights were modified." in str(error.value)


def test_gptq_layouts_still_accept_model_rotation():
    assert GPTQConfig(bits=4, group_size=128, rotation="hadamard").model_rotation == "hadamard"
    # RTN exports a GPTQ-layout checkpoint and its kernels read the rotation.
    assert RTNConfig(bits=8, rotation="hadamard").model_rotation == "hadamard"


def test_macaq_keeps_its_per_layer_rotation_field():
    # MACaQ reuses `rotation` for its own per-layer transform; the shared
    # model-wide rotation stays unset, so the gate must not fire.
    config = MacaqConfig(rotation="hadamard", offload_to_disk=False)

    assert config.model_rotation is None


def test_quantize_rejects_unsupported_rotation_backend_before_fusing(monkeypatch, tmp_path):
    def _must_not_run(*args, **kwargs):
        raise AssertionError("the model was fused or rotated before the backend gate")

    monkeypatch.setattr(base_module, "fuse_layer_norms", _must_not_run)
    monkeypatch.setattr(base_module, "rotate_model", _must_not_run)

    source = _tiny_source(tmp_path / "source")
    model = GPTQModel.load(
        str(source),
        quantize_config=GPTQConfig(
            bits=4, group_size=32, desc_act=False, rotation="hadamard", device="cpu"
        ),
        backend=BACKEND.GPTQ_TORCH,
        dtype=torch.bfloat16,
    )

    # MARLIN is a valid GPTQ backend but none of its kernels apply the online
    # Hadamard, so the rotation gate must refuse it.
    with pytest.raises(NotImplementedError, match="No weights were modified"):
        model.quantize(CALIBRATION, batch_size=1, backend=BACKEND.MARLIN)
