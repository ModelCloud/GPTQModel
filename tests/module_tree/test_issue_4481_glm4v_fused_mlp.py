# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""GLM-4V native fused MLP must be defused before strict plan resolution (#4481).

Transformers 5.22 builds `Glm4vTextMLP` with one fused `gate_up_proj` linear,
while `Glm4vGPTQ` declares `gate_proj`/`up_proj`. Defuser replaces the class
when it can patch it before construction, but a model built with
`loader.from_config()` (and the module-plan registry sweep) never gets that
patch, so the fused weight survives conversion and strict resolution raises on
`mlp.gate_proj` before the first MLP projection is quantized.
"""

import torch
import transformers

from gptqmodel.models.auto import MODEL_MAP
from gptqmodel.models.definitions.glm4v import Glm4vGPTQ
from gptqmodel.models.loader import _convert_model_with_defuser
from gptqmodel.models.local import LocalModelLoader
from gptqmodel.models.module_plan import ModulePlan
from gptqmodel.models.ports.native.auto import AutoConfig as NativeAutoConfig


def _auto_config(model_type, **kwargs):
    """Build the default config with the class the definition's loader expects."""
    loader = getattr(MODEL_MAP.get(model_type), "loader", None)
    if isinstance(loader, LocalModelLoader) and loader.config_class.startswith("gptqmodel.models.ports.native."):
        return NativeAutoConfig.for_model(model_type, **kwargs)
    return transformers.AutoConfig.for_model(model_type, **kwargs)


def _tiny_config():
    return _auto_config(
        "glm4v",
        text_config={
            "vocab_size": 128,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
        },
        vision_config={
            "depth": 1,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_heads": 4,
            "image_size": 28,
            "patch_size": 14,
            "out_hidden_size": 32,
        },
    )


def test_glm4v_fused_mlp_is_split_and_owned_by_the_plan():
    config = _tiny_config()
    model = Glm4vGPTQ.loader.from_config(config)
    layer = model.model.language_model.layers[0]
    mlp = layer.mlp
    assert mlp.__class__.__name__ == "Glm4vTextMLP"
    assert isinstance(mlp.gate_up_proj, torch.nn.Linear)

    hidden = torch.randn(2, 3, config.text_config.hidden_size)
    fused_output = mlp(hidden)
    fused_weight = mlp.gate_up_proj.weight.detach().clone()

    assert _convert_model_with_defuser(Glm4vGPTQ, model, cleanup_original=False)

    assert not hasattr(mlp, "gate_up_proj")
    half = fused_weight.shape[0] // 2
    assert mlp.gate_proj.weight.shape == mlp.up_proj.weight.shape == (half, fused_weight.shape[1])
    assert torch.equal(mlp.gate_proj.weight, fused_weight[:half])
    assert torch.equal(mlp.up_proj.weight, fused_weight[half:])
    # The fused class forward referenced `gate_up_proj`; the split instance must
    # stay runnable for calibration.
    assert torch.allclose(mlp(hidden), fused_output, atol=1e-6, rtol=1e-5)

    plan = ModulePlan.from_tree(Glm4vGPTQ.module_tree)
    for path in ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
        spec = plan.find(path)
        assert spec is not None, f"{path} is not declared"
        assert spec.quantize is True, (path, spec.flags)
    assert plan.find("mlp.gate_up_proj") is None

    # A second conversion pass (for example after a later Defuser version
    # already defused the class) must be a no-op.
    _convert_model_with_defuser(Glm4vGPTQ, model, cleanup_original=False)
    assert torch.equal(mlp.gate_proj.weight, fused_weight[:half])
