# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest

from gptqmodel.quantization import FORMAT, METHOD, AWQConfig, GGUFConfig, GPTQConfig
from gptqmodel.quantization.protocol import (
    MatchSpec,
    Rule,
    Stage,
    compile_plan_to_quantize_config,
    compile_protocol,
    compile_protocol_to_quantize_config,
    compile_protocol_yaml_text,
    compile_protocol_yaml_to_quantize_config,
)


def _python_protocol():
    return {
        "version": 2,
        "stages": [
            Stage(
                name="weight_only",
                rules=[
                    Rule(
                        match="*",
                        weight={
                            "quantize": {
                                "method": "gguf",
                                "bits": "q4_k_m",
                            },
                            "export": {
                                "format": "gguf",
                                "variant": "q_k_m",
                                "impl": "gguf_torch",
                            },
                        },
                    ),
                ],
            ),
        ],
    }


def _yaml_protocol() -> str:
    return """
version: 2
stages:
  - name: weight_only
    rules:
      - match: "*"
        weight:
          quantize:
            method: gguf
            bits: q4_k_m
          export:
            format: gguf
            variant: q_k_m
            impl: gguf_torch
"""


def _python_protocol_with_negative_match():
    return {
        "version": 2,
        "stages": [
            {
                "name": "weight_only",
                "rules": [
                    {
                        "match": ["*", r"-:.*layers\.2.*"],
                        "weight": {
                            "quantize": {
                                "method": "gguf",
                                "bits": "q4_k_m",
                            },
                            "export": {
                                "format": "gguf",
                                "variant": "q_k_m",
                                "impl": "gguf_torch",
                            },
                        },
                    },
                ],
            },
        ],
    }


def _yaml_protocol_with_negative_match() -> str:
    return r"""
version: 2
stages:
  - name: weight_only
    rules:
      - match:
          - "*"
          - '-:.*layers\.2.*'
        weight:
          quantize:
            method: gguf
            bits: q4_k_m
          export:
            format: gguf
            variant: q_k_m
            impl: gguf_torch
"""


def _python_gptq_protocol_with_negative_match():
    return {
        "version": 2,
        "stages": [
            {
                "name": "ptq",
                "rules": [
                    {
                        "match": ["*", r"-:.*layers\.2.*"],
                        "weight": {
                            "quantize": {
                                "method": "gptq",
                                "bits": 4,
                                "group_size": 128,
                                "sym": True,
                                "desc_act": False,
                            },
                            "export": {
                                "format": "gptq",
                                "variant": "gptq",
                                "impl": "marlin",
                            },
                        },
                    },
                ],
            },
        ],
    }


def _yaml_gptq_protocol_with_negative_match() -> str:
    return r"""
version: 2
stages:
  - name: ptq
    rules:
      - match:
          - "*"
          - '-:.*layers\.2.*'
        weight:
          quantize:
            method: gptq
            bits: 4
            group_size: 128
            sym: true
            desc_act: false
          export:
            format: gptq
            variant: gptq
            impl: marlin
"""


def _python_awq_protocol_with_negative_match():
    return {
        "version": 2,
        "stages": [
            {
                "name": "ptq",
                "rules": [
                    {
                        "match": ["*", r"-:.*layers\.2.*"],
                        "weight": {
                            "quantize": {
                                "method": "awq",
                                "bits": 4,
                                "group_size": 128,
                                "sym": True,
                                "desc_act": False,
                            },
                            "export": {
                                "format": "awq",
                                "variant": "gemm",
                                "impl": "gemm",
                            },
                        },
                    },
                ],
            },
        ],
    }


def _yaml_awq_protocol_with_negative_match() -> str:
    return r"""
version: 2
stages:
  - name: ptq
    rules:
      - match:
          - "*"
          - '-:.*layers\.2.*'
        weight:
          quantize:
            method: awq
            bits: 4
            group_size: 128
            sym: true
            desc_act: false
          export:
            format: awq
            variant: gemm
            impl: gemm
"""


def test_protocol_python_and_yaml_compile_to_same_execution_plan():
    python_plan = compile_protocol(_python_protocol())
    yaml_plan = compile_protocol_yaml_text(_yaml_protocol())

    assert python_plan == yaml_plan


def test_protocol_python_and_yaml_compile_to_same_gguf_config():
    python_cfg = compile_protocol_to_quantize_config(_python_protocol())
    yaml_cfg = compile_protocol_yaml_to_quantize_config(_yaml_protocol())

    assert isinstance(python_cfg, GGUFConfig)
    assert isinstance(yaml_cfg, GGUFConfig)
    assert python_cfg.to_dict() == yaml_cfg.to_dict()
    assert python_cfg.quant_method == METHOD.GGUF
    assert python_cfg.runtime_bits == "q4_k_m"
    assert python_cfg.format == "q_k_m"


def test_protocol_plan_compiles_to_expected_gguf_config():
    plan = compile_protocol(_python_protocol())
    cfg = compile_plan_to_quantize_config(plan)

    assert isinstance(cfg, GGUFConfig)
    assert cfg.quant_method == METHOD.GGUF
    assert cfg.bits == 4
    assert cfg.runtime_bits == "q4_k_m"
    assert cfg.format == "q_k_m"


def _gguf_protocol(bits, variant):
    export = {"format": "gguf"}
    if variant is not None:
        export["variant"] = variant
    return {
        "version": 2,
        "stages": [
            Stage(
                name="weight_only",
                rules=[
                    Rule(
                        match="*",
                        weight={
                            "quantize": {"method": "gguf", "bits": bits},
                            "export": export,
                        },
                    ),
                ],
            ),
        ],
    }


@pytest.mark.parametrize(
    ("bits", "variant"),
    [
        # The subtype exists at both widths, so only a real width comparison
        # can reject these. `bits` may be numeric or a full subtype alias.
        (4, "q8_0"),
        (4, "q6_k"),
        (6, "q4_k"),
        (8, "q4_0"),
        (4, "q5_k_m"),
        (8, "q4_k_m"),
        ("q8_0", "q4_0"),
        ("q4_k_m", "q4_0"),
    ],
)
def test_gguf_protocol_rejects_conflicting_bits_and_variant(bits, variant):
    with pytest.raises(ValueError, match="incompatible bits/export variant") as excinfo:
        compile_protocol_to_quantize_config(_gguf_protocol(bits, variant))

    message = str(excinfo.value)
    assert f"bits={bits}" in message
    assert f"export.variant={variant}" in message


@pytest.mark.parametrize(
    ("bits", "variant", "expected"),
    [
        # Documented payload: a full subtype in both fields must agree.
        ("q4_k_m", "q4_k_m", "q4_k_m"),
        (4, "q4_k_m", "q4_k_m"),
        ("q4_k_m", "q_k_m", "q4_k_m"),
        (4, "q_k_m", "q4_k_m"),
        (8, "q8_0", "q8_0"),
        ("q8_0", "q8_0", "q8_0"),
        (6, "q6_k", "q6_k"),
        ("q6_k", None, "q6_k"),
    ],
)
def test_gguf_protocol_accepts_consistent_bits_and_variant(bits, variant, expected):
    cfg = compile_protocol_to_quantize_config(_gguf_protocol(bits, variant))

    assert isinstance(cfg, GGUFConfig)
    assert cfg.runtime_bits.to_string() == expected


def test_documented_gguf_variant_q4_k_m_compiles():
    # docs/quantization_protocol.md example: export variant names the full
    # subtype rather than the projected `q_k_m` token.
    cfg = compile_protocol_to_quantize_config(
        {
            "version": 2,
            "stages": [
                {
                    "name": "weight_only",
                    "rules": [
                        {
                            "match": "*",
                            "weight": {
                                "quantize": {"method": "gguf", "bits": "q4_k_m"},
                                "export": {"format": "gguf", "variant": "q4_k_m"},
                            },
                        }
                    ],
                }
            ],
        }
    )

    assert cfg.runtime_bits.to_string() == "q4_k_m"
    assert cfg.format == "q_k_m"


def test_protocol_python_and_yaml_compile_to_same_negative_match_plan():
    python_plan = compile_protocol(_python_protocol_with_negative_match())
    yaml_plan = compile_protocol_yaml_text(_yaml_protocol_with_negative_match())

    assert python_plan == yaml_plan
    rule = python_plan.stages[0].rules[0]
    assert rule.match == (
        MatchSpec(pattern="*", include=True),
        MatchSpec(pattern=r".*layers\.2.*", include=False),
    )


def test_rule_match_supports_include_and_exclude_selectors():
    plan = compile_protocol(_python_protocol_with_negative_match())
    rule = plan.stages[0].rules[0]

    assert rule.matches("model.layers.0.self_attn.q_proj")
    assert not rule.matches("model.layers.2.self_attn.q_proj")


def test_negative_match_gguf_protocol_compiles_to_dynamic_skip_config():
    python_cfg = compile_protocol_to_quantize_config(_python_protocol_with_negative_match())
    yaml_cfg = compile_protocol_yaml_to_quantize_config(_yaml_protocol_with_negative_match())

    assert isinstance(python_cfg, GGUFConfig)
    assert isinstance(yaml_cfg, GGUFConfig)
    assert python_cfg.to_dict() == yaml_cfg.to_dict()
    assert python_cfg.quant_method == METHOD.GGUF
    assert python_cfg.runtime_bits == "q4_k_m"
    assert python_cfg.format == "q_k_m"
    assert python_cfg.dynamic == {r"-:.*(?:.*layers\.2.*)": {}}


def test_negative_match_gptq_protocol_compiles_to_dynamic_skip_config():
    python_cfg = compile_protocol_to_quantize_config(_python_gptq_protocol_with_negative_match())
    yaml_cfg = compile_protocol_yaml_to_quantize_config(_yaml_gptq_protocol_with_negative_match())

    assert isinstance(python_cfg, GPTQConfig)
    assert isinstance(yaml_cfg, GPTQConfig)
    assert type(python_cfg) is type(yaml_cfg)
    assert python_cfg.quant_method == METHOD.GPTQ
    assert yaml_cfg.quant_method == METHOD.GPTQ
    assert python_cfg.format == FORMAT.GPTQ
    assert yaml_cfg.format == FORMAT.GPTQ
    assert python_cfg.bits == 4
    assert yaml_cfg.bits == 4
    assert python_cfg.group_size == 128
    assert yaml_cfg.group_size == 128
    assert python_cfg.sym is True
    assert yaml_cfg.sym is True
    assert python_cfg.desc_act is False
    assert yaml_cfg.desc_act is False
    assert python_cfg.dynamic == {r"-:.*(?:.*layers\.2.*)": {}}
    assert yaml_cfg.dynamic == {r"-:.*(?:.*layers\.2.*)": {}}


def test_negative_match_awq_protocol_compiles_to_dynamic_skip_config():
    python_cfg = compile_protocol_to_quantize_config(_python_awq_protocol_with_negative_match())
    yaml_cfg = compile_protocol_yaml_to_quantize_config(_yaml_awq_protocol_with_negative_match())

    assert isinstance(python_cfg, AWQConfig)
    assert isinstance(yaml_cfg, AWQConfig)
    assert type(python_cfg) is type(yaml_cfg)
    assert python_cfg.quant_method == METHOD.AWQ
    assert yaml_cfg.quant_method == METHOD.AWQ
    assert python_cfg.format == FORMAT.GEMM
    assert yaml_cfg.format == FORMAT.GEMM
    assert python_cfg.bits == 4
    assert yaml_cfg.bits == 4
    assert python_cfg.group_size == 128
    assert yaml_cfg.group_size == 128
    assert python_cfg.sym is True
    assert yaml_cfg.sym is True
    assert python_cfg.dynamic == {r"-:.*(?:.*layers\.2.*)": {}}
    assert yaml_cfg.dynamic == {r"-:.*(?:.*layers\.2.*)": {}}


def _gptq_yaml_with_weight_quantize(extra: str = "") -> str:
    return (
        "version: 2\n"
        "stages:\n"
        "  - name: ptq\n"
        "    rules:\n"
        "      - match: \"*\"\n"
        "        weight:\n"
        "          quantize:\n"
        "            method: gptq\n"
        "            bits: 4\n"
        f"{extra}"
        "          export:\n"
        "            format: gptq\n"
    )


@pytest.mark.parametrize("target", ["input", "output", "kv_cache"])
def test_weight_rule_rejects_dropped_non_weight_targets(target):
    protocol = _gptq_yaml_with_weight_quantize(
        f"        {target}:\n"
        "          quantize: {method: fp8}\n"
    )

    with pytest.raises(NotImplementedError, match="only supports weight-target compilation") as excinfo:
        compile_protocol_yaml_to_quantize_config(protocol)

    assert f"`{target}`" in str(excinfo.value)


def test_weight_rule_rejects_input_and_kv_cache_together():
    protocol = _gptq_yaml_with_weight_quantize(
        "        input:\n"
        "          quantize: {method: mxfp4, mode: dynamic}\n"
        "        kv_cache:\n"
        "          quantize: {method: fp8}\n"
    )

    with pytest.raises(NotImplementedError, match="only supports weight-target compilation") as excinfo:
        compile_protocol_yaml_to_quantize_config(protocol)

    message = str(excinfo.value)
    assert "`input`" in message and "`kv_cache`" in message


@pytest.mark.parametrize(
    ("extra", "unknown"),
    [
        ("            act_order: true\n", "act_order"),
        ("            damp_percent: 0.1\n", "damp_percent"),
        ("            symmetric: false\n", "symmetric"),
        ("            group-size: 32\n", "group-size"),
        ("            mse: true\n", "mse"),
        ("            true_sequential: false\n", "true_sequential"),
    ],
)
def test_gptq_weight_quantize_rejects_unknown_arguments(extra, unknown):
    with pytest.raises(ValueError, match="unsupported `weight.quantize` argument") as excinfo:
        compile_protocol_yaml_to_quantize_config(_gptq_yaml_with_weight_quantize(extra))

    message = str(excinfo.value)
    assert f"`{unknown}`" in message
    assert "Supported keys:" in message
    for key in ("bits", "group_size", "sym", "desc_act", "act_group_aware"):
        assert f"`{key}`" in message


def test_gptq_weight_quantize_accepts_every_supported_argument():
    protocol = _gptq_yaml_with_weight_quantize(
        "            group_size: 32\n"
        "            sym: false\n"
        "            desc_act: false\n"
        "            act_group_aware: true\n"
    )

    cfg = compile_protocol_yaml_to_quantize_config(protocol)

    assert isinstance(cfg, GPTQConfig)
    assert cfg.bits == 4
    assert cfg.group_size == 32
    assert cfg.sym is False
    assert cfg.desc_act is False
    assert cfg.act_group_aware is True


def test_gguf_weight_quantize_rejects_unknown_arguments():
    protocol = """
version: 2
stages:
  - name: weight_only
    rules:
      - match: "*"
        weight:
          quantize:
            method: gguf
            bits: q4_k_m
            group_size: 32
"""

    with pytest.raises(ValueError, match="unsupported `weight.quantize` argument") as excinfo:
        compile_protocol_yaml_to_quantize_config(protocol)

    message = str(excinfo.value)
    assert "`group_size`" in message
    assert "Supported keys: `bits`" in message
