# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

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
    assert python_cfg.dynamic == {r"-:.*layers\.2.*": {}}


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
    assert python_cfg.dynamic == {r"-:.*layers\.2.*": {}}
    assert yaml_cfg.dynamic == {r"-:.*layers\.2.*": {}}


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
    assert python_cfg.dynamic == {r"-:.*layers\.2.*": {}}
    assert yaml_cfg.dynamic == {r"-:.*layers\.2.*": {}}
