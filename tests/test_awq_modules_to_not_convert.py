import json

from gptqmodel.quantization.config import AWQConfig, BaseQuantizeConfig


def test_awq_modules_to_not_convert_excluded_at_construction_and_reload():
    target = "model.layers.0.mlp.down_proj"
    other = "model.layers.0.mlp.up_proj"

    cfg = AWQConfig(bits=4, group_size=128, modules_to_not_convert=["down_proj"])
    assert cfg.dynamic is not None
    assert cfg.resolve_module_config(target).excluded is True
    assert cfg.resolve_module_config(other).excluded is False

    reloaded = BaseQuantizeConfig.from_quant_config(json.loads(json.dumps(cfg.to_dict())))
    assert reloaded.resolve_module_config(target).excluded is True
    assert reloaded.resolve_module_config(other).excluded is False
