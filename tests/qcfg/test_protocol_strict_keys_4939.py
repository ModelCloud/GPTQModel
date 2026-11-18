# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""The protocol compiler must fail closed on values and keys it cannot honor (#4939).

#4596 made the compiler reject unknown ``weight.quantize`` arguments, but two
ways of dropping user intent survived:

* ``sym``/``desc_act``/``act_group_aware``/``stop`` were coerced with
  ``bool(...)``, so a quoted YAML/JSON ``"false"`` became ``True`` -- the strict
  bool check added in #4668 never saw the string;
* unknown keys above ``weight.quantize`` were read with ``data.get(...)``, so a
  misspelled ``export``/``variant``/``prepare``/``stop`` silently compiled to
  the default GPTQ v1 config.

The tests below assert rejection (not just the compiled value), because every
one of these paths used to "succeed" with a config that differed from the plan.
"""

import pytest

from gptqmodel.quantization.config import FORMAT, GGUFConfig, GPTQConfig
from gptqmodel.quantization.protocol import (
    Rule,
    TargetSpec,
    compile_protocol,
    compile_protocol_to_quantize_config,
    compile_protocol_yaml_text,
    compile_protocol_yaml_to_quantize_config,
)


_QUANTIZE = "method: gptq, bits: 4, group_size: 128"


def _protocol(
    *,
    quantize: str = _QUANTIZE,
    target_extra: str = "",
    rule_extra: str = "",
    stage_extra: str = "",
) -> str:
    """Build a one-stage protocol; the extras land at the documented indents."""

    def block(text: str, indent: int) -> str:
        if not text.strip():
            return ""
        return "".join(" " * indent + line + "\n" for line in text.strip("\n").splitlines())

    return (
        "version: 2\n"
        "stages:\n"
        "  - name: ptq\n"
        f"{block(stage_extra, 4)}"
        "    rules:\n"
        '      - match: "*"\n'
        f"{block(rule_extra, 8)}"
        "        weight:\n"
        f"          quantize: {{{quantize}}}\n"
        f"{block(target_extra, 10)}"
    )


# --- booleans: reuse the #4668 strict check, never bool() -----------------


@pytest.mark.parametrize(
    ("quantize", "label"),
    [
        (_QUANTIZE + ', sym: "false"', "weight.quantize.sym"),
        (_QUANTIZE + ', desc_act: "no"', "weight.quantize.desc_act"),
        (_QUANTIZE + ', act_group_aware: "true"', "weight.quantize.act_group_aware"),
        (_QUANTIZE + ", sym: 0", "weight.quantize.sym"),
        (_QUANTIZE + ", sym: \"0\"", "weight.quantize.sym"),
        (_QUANTIZE + ", desc_act: 1", "weight.quantize.desc_act"),
    ],
)
def test_quoted_and_int_booleans_are_rejected(quantize, label):
    with pytest.raises(TypeError, match="must be a bool") as excinfo:
        compile_protocol_yaml_to_quantize_config(_protocol(quantize=quantize))

    message = str(excinfo.value)
    assert f"`{label}`" in message
    # Actionable: name the quoted-value trap instead of echoing a bare type error.
    assert "quoted" in message.lower() or "string" in message.lower()


def test_quoted_rule_stop_is_rejected():
    with pytest.raises(TypeError, match="must be a bool") as excinfo:
        compile_protocol_yaml_to_quantize_config(
            _protocol(rule_extra='stop: "false"')
        )

    assert "`stop`" in str(excinfo.value)


def test_rule_dataclass_stop_is_checked_too():
    with pytest.raises(TypeError, match="must be a bool"):
        compile_protocol(
            {
                "version": 2,
                "stages": [
                    {
                        "name": "ptq",
                        "rules": [Rule(match="*", stop="false")],
                    }
                ],
            }
        )


def test_real_booleans_still_compile():
    cfg = compile_protocol_yaml_to_quantize_config(
        _protocol(quantize=_QUANTIZE + ", sym: false, desc_act: false, act_group_aware: true")
    )

    assert isinstance(cfg, GPTQConfig)
    assert cfg.sym is False
    assert cfg.desc_act is False
    assert cfg.act_group_aware is True


# --- unknown keys at every level above weight.quantize --------------------


def _assert_unknown_key_error(excinfo, key: str, accepted: tuple[str, ...]):
    message = str(excinfo.value)
    assert f"`{key}`" in message, message
    for allowed in accepted:
        assert f"`{allowed}`" in message, (allowed, message)
    return message


def test_unknown_target_key_is_rejected():
    with pytest.raises(ValueError, match="unknown") as excinfo:
        compile_protocol_yaml_to_quantize_config(
            _protocol(target_extra="exprot: {format: gptq, variant: gptq_v2}")
        )

    message = _assert_unknown_key_error(
        excinfo, "exprot", ("mode", "prepare", "quantize", "export")
    )
    assert "did you mean `export`?" in message


def test_misspelled_export_field_does_not_silently_yield_gptq_v1():
    with pytest.raises(ValueError, match="unknown") as excinfo:
        compile_protocol_yaml_to_quantize_config(
            _protocol(target_extra="export: {format: gptq, varient: gptq_v2}")
        )

    message = _assert_unknown_key_error(
        excinfo, "varient", ("format", "variant", "impl", "version", "options")
    )
    assert "did you mean `variant`?" in message


def test_unknown_rule_key_is_rejected():
    with pytest.raises(ValueError, match="unknown") as excinfo:
        compile_protocol_yaml_to_quantize_config(_protocol(rule_extra="stopp: true"))

    message = _assert_unknown_key_error(
        excinfo,
        "stopp",
        ("match", "aliases", "actions", "stop", "weight", "input", "output", "kv_cache"),
    )
    assert "did you mean `stop`?" in message


def test_misspelled_prepare_is_rejected_instead_of_dropped():
    with pytest.raises(ValueError, match="unknown") as excinfo:
        compile_protocol_yaml_to_quantize_config(
            _protocol(target_extra="prepair: [{method: smooth.mad}]")
        )

    message = _assert_unknown_key_error(
        excinfo, "prepair", ("mode", "prepare", "quantize", "export")
    )
    assert "did you mean `prepare`?" in message


def test_unknown_stage_key_is_rejected():
    with pytest.raises(ValueError, match="unknown") as excinfo:
        compile_protocol_yaml_text(
            "version: 2\n"
            "stages:\n"
            "  - name: ptq\n"
            "    rulez: []\n"
            "    rules:\n"
            '      - match: "*"\n'
            "        weight:\n"
            "          quantize: {method: gptq, bits: 4}\n"
        )

    message = _assert_unknown_key_error(excinfo, "rulez", ("name", "rules"))
    assert "did you mean `rules`?" in message


def test_unknown_root_key_is_rejected():
    with pytest.raises(ValueError, match="unknown") as excinfo:
        compile_protocol_yaml_text(
            "version: 2\n"
            "stagez:\n"
            "  - name: ptq\n"
            "stages:\n"
            "  - name: ptq\n"
            "    rules:\n"
            '      - match: "*"\n'
            "        weight:\n"
            "          quantize: {method: gptq, bits: 4}\n"
        )

    _assert_unknown_key_error(excinfo, "stagez", ("version", "stages"))


def test_misspelled_weight_quantize_key_suggests_the_real_one():
    with pytest.raises(ValueError, match="unsupported") as excinfo:
        compile_protocol_yaml_to_quantize_config(
            _protocol(quantize=_QUANTIZE + ", desc_actt: true")
        )

    message = str(excinfo.value)
    assert "`desc_actt`" in message
    assert "did you mean `desc_act`?" in message


# --- export fields the compiler never reads -------------------------------


def test_export_options_are_rejected_not_dropped():
    with pytest.raises(NotImplementedError, match="options") as excinfo:
        compile_protocol_yaml_to_quantize_config(
            _protocol(target_extra="export: {format: gptq, options: {pack_dtype: int16}}")
        )

    assert "pack_dtype" in str(excinfo.value)


def test_export_version_is_rejected_not_dropped():
    with pytest.raises(NotImplementedError, match="version"):
        compile_protocol_yaml_to_quantize_config(
            _protocol(target_extra="export: {format: gptq, version: 99}")
        )


# --- valid protocols are unchanged ----------------------------------------


def test_valid_protocol_still_compiles_to_the_same_config():
    cfg = compile_protocol_yaml_to_quantize_config(
        _protocol(
            quantize=(
                "method: gptq, bits: 4, group_size: 32, sym: false, "
                "desc_act: false, act_group_aware: true"
            ),
            target_extra=(
                "export: {format: gptq, variant: gptq_v2, impl: marlin}\n"
                "mode: merge"
            ),
        )
    )

    assert isinstance(cfg, GPTQConfig)
    assert cfg.quant_method.value == "gptq"
    assert cfg.format == FORMAT.GPTQ_V2
    assert cfg.bits == 4
    assert cfg.group_size == 32
    assert cfg.sym is False
    assert cfg.desc_act is False
    assert cfg.act_group_aware is True
    assert cfg.dynamic is None


def test_python_dataclass_protocol_still_compiles():
    # The allow-lists must not reject the typed dataclass path.
    cfg = compile_protocol_to_quantize_config(
        {
            "version": 2,
            "stages": [
                {
                    "name": "ptq",
                    "rules": [
                        Rule(
                            match="*",
                            stop=False,
                            weight=TargetSpec(
                                mode="merge",
                                quantize={"method": "gptq", "bits": 4, "group_size": 128},
                            ),
                        )
                    ],
                }
            ],
        }
    )

    assert isinstance(cfg, GPTQConfig)
    assert cfg.bits == 4
    assert cfg.format == FORMAT.GPTQ


def test_gguf_export_options_are_rejected_too():
    protocol = """
version: 2
stages:
  - name: weight_only
    rules:
      - match: "*"
        weight:
          quantize: {method: gguf, bits: q4_k_m}
          export: {format: gguf, options: {pack_dtype: int16}}
"""
    with pytest.raises(NotImplementedError, match="options") as excinfo:
        compile_protocol_yaml_to_quantize_config(protocol)

    assert "pack_dtype" in str(excinfo.value)


def _gguf_export_protocol(*, bits: str, variant: str) -> str:
    return (
        "version: 2\n"
        "stages:\n"
        "  - name: weight_only\n"
        "    rules:\n"
        '      - match: "*"\n'
        "        weight:\n"
        f"          quantize: {{method: gguf, bits: {bits}}}\n"
        f"          export: {{format: gguf, variant: {variant}}}\n"
    )


def test_gguf_variant_conflicting_with_bits_is_rejected():
    with pytest.raises(ValueError, match="incompatible bits/export variant") as excinfo:
        compile_protocol_yaml_to_quantize_config(
            _gguf_export_protocol(bits="4", variant="q8_0")
        )

    message = str(excinfo.value)
    assert "bits=4" in message
    assert "export.variant=q8_0" in message


def test_documented_gguf_variant_q4_k_m_is_accepted():
    cfg = compile_protocol_yaml_to_quantize_config(
        _gguf_export_protocol(bits="q4_k_m", variant="q4_k_m")
    )

    assert isinstance(cfg, GGUFConfig)
    assert cfg.runtime_bits.to_string() == "q4_k_m"


def test_export_options_survive_plan_compilation_only():
    # A plan is a faithful parse of the document; only the config compiler,
    # which has no consumer for these fields, refuses them.
    plan = compile_protocol_yaml_text(_protocol(target_extra="export: {format: gptq, options: {pack_dtype: int16}}"))
    export = plan.stages[0].rules[0].weight.export
    assert export.options == {"pack_dtype": "int16"}
