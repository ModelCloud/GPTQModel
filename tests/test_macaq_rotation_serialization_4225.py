# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""#4225: MACaQ must not serialise its per-layer transform into the shared key.

Older readers (`models/loader.py`, `utils/hf_load.py`) interpret a truthy
top-level `rotation` as a model-wide rotation and reject every non-GPTQ
checkpoint. MACaQ's own `rotation` is per-layer, so new checkpoints store it
under `macaq_rotation` and leave the shared key empty.
"""

import os


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from gptqmodel.quantization import MacaqConfig, QuantizeConfig
from gptqmodel.quantization.config import resolve_model_rotation


def test_default_macaq_keeps_the_model_wide_rotation_key_empty():
    config = MacaqConfig(offload_to_disk=False)
    payload = config.to_dict()

    # The pre-9dad249 loader only checks `if qcfg.rotation:`; a missing or
    # null value is what keeps it loading new checkpoints.
    assert not payload.get("rotation")
    assert payload["macaq_rotation"] == "hadamard"
    assert resolve_model_rotation(config) is None


def test_default_macaq_payload_round_trips_to_hadamard():
    payload = MacaqConfig(offload_to_disk=False).to_dict()

    restored = QuantizeConfig.from_quant_config(payload)
    assert isinstance(restored, MacaqConfig)
    assert restored.rotation == "hadamard"
    assert resolve_model_rotation(restored) is None

    # `QuantizeConfig(**config.to_dict())` must accept the serialized form.
    factory = QuantizeConfig(**payload)
    assert isinstance(factory, MacaqConfig)
    assert factory.rotation == "hadamard"


def test_macaq_rotation_none_uses_the_per_layer_key():
    config = MacaqConfig(rotation="none", activation_precision="w4a16", offload_to_disk=False)
    payload = config.to_dict()

    assert not payload.get("rotation")
    assert payload["macaq_rotation"] == "none"
    assert QuantizeConfig.from_quant_config(payload).rotation == "none"


def test_head_checkpoints_with_the_shared_rotation_key_still_load():
    # 9dad249..HEAD wrote MACaQ's transform into `rotation`; keep those
    # checkpoints loadable at HEAD.
    payload = {
        "method": "macaq",
        "format": "mxfp4_macaq",
        "bits": 4,
        "rotation": "none",
        "sym": True,
    }
    assert QuantizeConfig.from_quant_config(payload).rotation == "none"

    payload["rotation"] = None
    assert QuantizeConfig.from_quant_config(payload).rotation == "hadamard"
