import json
import pcre

from gptqmodel.quantization.config import BaseQuantizeConfig, GPTQConfig


def test_runtime_added_negative_dynamic_rule_takes_precedence_and_serializes_first():
    cfg = GPTQConfig(bits=4, dynamic={r".*\.experts\..*": {"bits": 8}})
    target = "model.layers.3.mlp.experts.17.down_proj"
    other = "model.layers.3.mlp.experts.18.down_proj"

    cfg.dynamic[f"-:{pcre.escape(target)}"] = {}

    assert cfg.dynamic_get(target) is False
    assert cfg.dynamic_get(other) == {"bits": 8}

    serialized = json.loads(json.dumps(cfg.to_dict()))
    keys = list(serialized["dynamic"])
    assert keys[0].startswith("-:")

    reloaded = BaseQuantizeConfig.from_quant_config(serialized)
    assert reloaded.dynamic_get(target) is False
    assert reloaded.dynamic_get(other) == {"bits": 8}
