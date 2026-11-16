import json

from gptqmodel.quantization.config import AWQConfig, BaseQuantizeConfig


K_PROJ = "model.layers.0.self_attn.k_proj"
Q_PROJ = "model.layers.0.self_attn.q_proj"
DOWN_PROJ = "model.layers.0.mlp.down_proj"


def test_assignment_after_construction_folds_into_dynamic():
    cfg = AWQConfig(bits=4, group_size=128)
    cfg.modules_to_not_convert = ["k_proj"]

    assert any(key.startswith("-:") for key in (cfg.dynamic or {}))
    assert cfg.resolve_module_config(K_PROJ).excluded is True
    assert cfg.resolve_module_config(Q_PROJ).excluded is False
    assert cfg.excluded_module_patterns() == ["k_proj"]


def test_assignment_after_construction_round_trips():
    cfg = AWQConfig(bits=4, group_size=128)
    cfg.modules_to_not_convert = ["k_proj"]

    payload = json.loads(json.dumps(cfg.to_dict()))
    assert payload["modules_to_not_convert"] == ["k_proj"]

    reloaded = BaseQuantizeConfig.from_quant_config(payload)
    assert reloaded.resolve_module_config(K_PROJ).excluded is True
    assert reloaded.resolve_module_config(Q_PROJ).excluded is False


def test_dynamic_reassignment_without_negative_rule_drops_claim():
    cfg = AWQConfig(bits=4, group_size=128, modules_to_not_convert=["down_proj"])
    cfg.dynamic = {"+:.*q_proj": {"bits": 8}}

    # The exclusion rule is gone, so quantization and the saved config must not
    # claim the module as excluded.
    assert cfg.resolve_module_config(DOWN_PROJ).excluded is False
    assert cfg.excluded_module_patterns() == []
    assert "modules_to_not_convert" not in cfg.to_dict()


def test_shrinking_declaration_removes_stale_rule():
    cfg = AWQConfig(bits=4, group_size=128, modules_to_not_convert=["down_proj", "k_proj"])
    cfg.modules_to_not_convert = ["down_proj"]

    assert cfg.resolve_module_config(K_PROJ).excluded is False
    assert cfg.resolve_module_config(DOWN_PROJ).excluded is True
    assert cfg.excluded_module_patterns() == ["down_proj"]


def test_clearing_declaration_removes_all_rules():
    cfg = AWQConfig(bits=4, group_size=128, modules_to_not_convert=["k_proj"])
    cfg.modules_to_not_convert = []

    assert cfg.resolve_module_config(K_PROJ).excluded is False
    assert cfg.dynamic in (None, {})
    assert cfg.excluded_module_patterns() == []


def test_constructor_declaration_is_unchanged():
    cfg = AWQConfig(bits=4, group_size=128, modules_to_not_convert=["down_proj"])

    assert cfg.resolve_module_config(DOWN_PROJ).excluded is True
    assert cfg.resolve_module_config(K_PROJ).excluded is False
    assert cfg.excluded_module_patterns() == ["down_proj"]
