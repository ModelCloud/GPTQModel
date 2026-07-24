# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import torch

from gptqmodel.looper.analysis_processor import AnalysisProcessor
from gptqmodel.models.definitions.qwen3 import Qwen3QModel
from gptqmodel.quantization import (
    AnalysisConfig,
    AnalysisSelection,
    GPTQConfig,
    QuantizationAnalyzer,
    QuantizeConfig,
    apply_analysis_plan,
    report_to_json,
)


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(8, 8, bias=False)
        self.down_proj = torch.nn.Linear(8, 8, bias=False)


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(16, 8)
        self.layers = torch.nn.ModuleList([_Block()])
        self.lm_head = torch.nn.Linear(8, 16, bias=False)
        self.lm_head.weight = self.embed_tokens.weight
        with torch.no_grad():
            self.layers[0].down_proj.weight[3, 5] = 1000.0

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head


def test_granular_analysis_covers_endpoints_and_bounded_regions():
    qcfg = GPTQConfig(bits=4, group_size=4)
    config = AnalysisConfig(
        top_k_regions=9,
        regions_per_module=2,
        recommendation_percentile=50,
        min_recommendation_risk=0,
    )
    report = QuantizationAnalyzer(qcfg, config).analyze_model(_TinyModel())

    records = {record["module"]: record for record in report["records"]}
    assert records["embed_tokens"]["role"] == "input_embedding"
    assert records["lm_head"]["role"] == "lm_head"
    assert records["lm_head"]["tied_to"] == "embed_tokens"
    assert "embedding_row" in {region["kind"] for region in records["embed_tokens"]["regions"]}
    assert "vocab_row" in {region["kind"] for region in records["lm_head"]["regions"]}
    assert records["layers.0.q_proj"]["role"] == "attention_q"
    assert records["layers.0.down_proj"]["role"] == "mlp_down"
    assert len(report["regions"]) <= 9
    assert {"output_channel", "input_feature", "group"} <= {
        region["kind"] for region in records["layers.0.down_proj"]["regions"]
    }
    assert report["summary"]["analyzed_modules"] == 4
    assert json.loads(report_to_json(report))["schema_version"] == "1.0"
    assert "## Executive summary" in report["markdown"]
    assert "## Direct recommendations and definition-group companions" in report["markdown"]
    assert "## Exact localized regions" in report["markdown"]
    assert "## Index semantics" in report["markdown"]
    assert "## Proposed quantizer mapping" in report["markdown"]
    assert "## Generated artifacts" in report["markdown"]
    assert "## Interpretation and limitations" in report["markdown"]
    assert "`embed_tokens`" in report["markdown"]
    assert "token row" in report["markdown"]
    assert "No `+1` conversion" in report["markdown"]


def test_extended_analysis_config_roundtrip():
    analysis = AnalysisConfig(
        top_k_regions=17,
        regions_per_module=3,
        include_endpoints=False,
        recommendation_percentile=90,
        min_recommendation_risk=35,
        promotion_bits=6,
        promotion_group_size=16,
        fusion_profile="none",
        max_chunk_values=4096,
        max_sample_values=1024,
    )
    restored = QuantizeConfig.from_quant_config(
        GPTQConfig(bits=4, group_size=128, preprocessors=[analysis]).to_dict()
    )
    restored_analysis = restored.preprocessors[0]

    assert restored_analysis.top_k_regions == 17
    assert restored_analysis.regions_per_module == 3
    assert restored_analysis.include_endpoints is False
    assert restored_analysis.recommendation_percentile == 90
    assert restored_analysis.min_recommendation_risk == 35
    assert restored_analysis.promotion_bits == 6
    assert restored_analysis.promotion_group_size == 16
    assert restored_analysis.fusion_profile == "none"
    assert restored_analysis.max_chunk_values == 4096
    assert restored_analysis.max_sample_values == 1024


def test_selection_and_tail_group_analysis():
    qcfg = GPTQConfig(bits=4, group_size=3, sym=False)
    report = QuantizationAnalyzer(qcfg).analyze_model(
        _TinyModel(),
        selection=AnalysisSelection(module_pattern=r"layers\.0\.", max_modules=1),
    )

    assert report["summary"]["analyzed_modules"] == 1
    record = report["records"][0]
    assert record["num_blocks"] == 8 * 3
    assert record["effective_group_size"] == 3
    assert all(region.get("input_end", 0) <= 8 for region in record["regions"])


def test_analysis_plan_application_is_explicit_and_preserves_user_rules():
    pattern = r"+:^layers\.0\.down_proj$"
    qcfg = GPTQConfig(
        bits=4,
        group_size=128,
        dynamic={pattern: {"bits": 6}},
    )
    config = AnalysisConfig(
        recommendation_percentile=1,
        min_recommendation_risk=0,
        promotion_bits=8,
        promotion_group_size=32,
    )
    report = QuantizationAnalyzer(qcfg, config).analyze_model(
        _TinyModel(),
        selection=AnalysisSelection(module_pattern=r"layers\.0\.down_proj"),
    )

    original_dynamic = dict(qcfg.dynamic)
    planned, merge = apply_analysis_plan(qcfg, report["plan"])

    assert qcfg.dynamic == original_dynamic
    assert planned.dynamic[pattern] == {"bits": 6}
    assert not merge["applied"]
    assert merge["conflicts"][0]["module"] == "layers.0.down_proj"


def test_inline_processor_streams_meta_shell_weight_from_checkpoint_source():
    layer = torch.nn.Module()
    layer.q_proj = torch.nn.Linear(8, 4, bias=False, device="meta")
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([layer])
    source_weight = torch.linspace(-1, 1, 32).reshape(4, 8)

    class _Turtle:
        def checkpoint_tensors_for_submodule(self, *, target_model, target_submodule, recurse):
            assert target_model is model
            assert target_submodule is layer.q_proj
            assert recurse is False
            return {"weight": source_weight}

    qcfg = GPTQConfig(
        bits=4,
        group_size=4,
        preprocessors=[AnalysisConfig(emit_markdown=False, emit_json=False)],
    )
    processor = AnalysisProcessor(qcfg=qcfg, tokenizer=None)
    processor.analyze_model(
        layers=[layer],
        layer_modules=[["q_proj"]],
        layers_prefix="model.layers",
        model=model,
        gptq_model=SimpleNamespace(turtle_model=_Turtle()),
    )

    assert len(processor.records) == 1
    assert processor.records[0]["module"] == "model.layers.0.q_proj"
    assert processor.records[0]["source_device"] == "cpu"
    assert layer.q_proj.weight.device.type == "meta"


def test_plan_expands_qkv_fusion_companions_without_calling_them_direct_outliers():
    class _Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(8, 8, bias=False)
            self.k_proj = torch.nn.Linear(8, 8, bias=False)
            self.v_proj = torch.nn.Linear(8, 8, bias=False)
            with torch.no_grad():
                self.q_proj.weight.fill_(1.0e-3)
                self.q_proj.weight[0, 0] = 100.0
                self.k_proj.weight.zero_()
                self.v_proj.weight.zero_()

    class _Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = _Attention()

    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([_Layer()])
    qcfg = GPTQConfig(bits=4, group_size=4)
    config = AnalysisConfig(
        recommendation_percentile=1,
        min_recommendation_risk=30,
        fusion_profile="model_definition",
    )
    module_groups = Qwen3QModel.simple_layer_modules(
        model_config=SimpleNamespace(),
        quantize_config=qcfg,
    )
    report = QuantizationAnalyzer(
        qcfg,
        config,
        module_groups=module_groups,
        module_group_source="gptqmodel.models.definitions.qwen3.Qwen3QModel.module_tree",
    ).analyze_model(model)
    recommendations = {item["module"]: item for item in report["plan"]["recommendations"]}

    assert recommendations["layers.0.self_attn.q_proj"]["action"] == "promote_module"
    assert recommendations["layers.0.self_attn.k_proj"]["action"] == "promote_fusion_companion"
    assert recommendations["layers.0.self_attn.v_proj"]["action"] == "promote_fusion_companion"
    assert recommendations["layers.0.self_attn.k_proj"]["triggered_by"] == [
        "layers.0.self_attn.q_proj"
    ]
    assert len(report["plan"]["dynamic"]) == 3
    assert report["summary"]["flagged_modules"] == 1
    assert report["summary"]["fusion_companions"] == 2
    group = report["plan"]["fusion_groups"][0]
    assert group["fusibility_basis"] == "gptqmodel_model_definition_module_group"
    assert group["definition_group"] == [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
    ]
    assert group["members"] == [
        "layers.0.self_attn.q_proj",
        "layers.0.self_attn.k_proj",
        "layers.0.self_attn.v_proj",
    ]
    assert "Qwen3QModel.module_tree" in group["group_source"]
    assert "## GPTQModel definition-group closure" in report["markdown"]

    unfused_report = QuantizationAnalyzer(
        qcfg,
        AnalysisConfig(
            recommendation_percentile=1,
            min_recommendation_risk=30,
            fusion_profile="none",
        ),
    ).analyze_model(model)
    assert len(unfused_report["plan"]["dynamic"]) == 1
    assert unfused_report["summary"]["fusion_companions"] == 0


def test_qwen3_gate_up_definition_group_drives_companion_closure():
    class _MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = torch.nn.Linear(8, 8, bias=False)
            self.up_proj = torch.nn.Linear(8, 8, bias=False)
            self.down_proj = torch.nn.Linear(8, 8, bias=False)
            with torch.no_grad():
                self.gate_proj.weight.fill_(1.0e-3)
                self.gate_proj.weight[0, 0] = 100.0
                self.up_proj.weight.zero_()
                self.down_proj.weight.zero_()

    class _Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = _MLP()

    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([_Layer()])
    qcfg = GPTQConfig(bits=4, group_size=4)
    module_groups = Qwen3QModel.simple_layer_modules(
        model_config=SimpleNamespace(),
        quantize_config=qcfg,
    )
    report = QuantizationAnalyzer(
        qcfg,
        AnalysisConfig(
            recommendation_percentile=1,
            min_recommendation_risk=30,
            fusion_profile="model_definition",
        ),
        module_groups=module_groups,
        module_group_source="gptqmodel.models.definitions.qwen3.Qwen3QModel.module_tree",
    ).analyze_model(model)
    recommendations = {item["module"]: item for item in report["plan"]["recommendations"]}

    assert recommendations["layers.0.mlp.gate_proj"]["action"] == "promote_module"
    assert recommendations["layers.0.mlp.up_proj"]["action"] == "promote_fusion_companion"
    assert "layers.0.mlp.down_proj" not in recommendations
    group = report["plan"]["fusion_groups"][0]
    assert group["kind"] == "gate_up_projection_group"
    assert group["definition_group_index"] == 2
    assert group["definition_group"] == ["mlp.gate_proj", "mlp.up_proj"]
    assert group["companions"] == ["layers.0.mlp.up_proj"]
