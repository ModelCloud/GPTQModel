from types import SimpleNamespace

import torch

from gptqmodel.looper.analysis_processor import AnalysisProcessor
from gptqmodel.quantization import AnalysisConfig, GPTQConfig, QuantizeConfig


class _TinyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.clean = torch.nn.Linear(8, 4, bias=False)
        self.outlier = torch.nn.Linear(8, 4, bias=False)
        with torch.no_grad():
            self.clean.weight.copy_(
                torch.tensor(
                    [
                        [-0.20, -0.15, -0.10, -0.05, 0.05, 0.10, 0.15, 0.20],
                        [-0.18, -0.12, -0.08, -0.04, 0.04, 0.08, 0.12, 0.18],
                        [-0.16, -0.11, -0.06, -0.03, 0.03, 0.06, 0.11, 0.16],
                        [-0.14, -0.09, -0.05, -0.02, 0.02, 0.05, 0.09, 0.14],
                    ]
                )
            )
            self.outlier.weight.fill_(1.0e-3)
            self.outlier.weight[0, 0] = 100.0


def test_analysis_config_roundtrip():
    cfg = GPTQConfig(
        bits=4,
        group_size=4,
        preprocessors=[
            AnalysisConfig(
                top_k=7,
                emit_markdown=False,
                emit_json=False,
                bad_block_rel_rmse_threshold=0.2,
            )
        ],
    )

    payload = cfg.to_dict()
    preprocessors = payload["meta"]["preprocessors"]
    assert preprocessors == [
        {
            "code": "analysis",
            "top_k": 7,
            "emit_markdown": False,
            "emit_json": False,
            "bad_block_rel_rmse_threshold": 0.2,
        }
    ]

    parsed = QuantizeConfig.from_quant_config(payload)
    assert isinstance(parsed.preprocessors[0], AnalysisConfig)
    assert parsed.preprocessors[0].top_k == 7
    assert parsed.preprocessors[0].emit_markdown is False
    assert parsed.preprocessors[0].emit_json is False
    assert parsed.preprocessors[0].bad_block_rel_rmse_threshold == 0.2


def test_analysis_processor_ranks_outlier_module_as_less_quantizable():
    qcfg = GPTQConfig(
        bits=4,
        group_size=4,
        preprocessors=[AnalysisConfig(top_k=2, emit_markdown=False, emit_json=False)],
    )
    processor = AnalysisProcessor(qcfg=qcfg, tokenizer=None)
    layer = _TinyLayer()

    processor.analyze_model(
        layers=[layer],
        layer_modules=[["clean", "outlier"]],
        layers_prefix="model.layers",
    )
    model = SimpleNamespace()
    processor.finalize(model=model)

    records = {record["name"]: record for record in model.quantize_analysis["records"]}
    assert records["outlier"]["quant_score"] < records["clean"]["quant_score"]
    assert records["outlier"]["max_to_median_abs"] > records["clean"]["max_to_median_abs"]
    assert "quant_score" in model.quantize_analysis["markdown"]


def test_analysis_processor_respects_dynamic_quant_config():
    qcfg = GPTQConfig(
        bits=4,
        group_size=4,
        dynamic={
            "-:model\\.layers\\.0\\.outlier": {},
            "+:model\\.layers\\.0\\.clean": {"bits": 8, "group_size": -1, "sym": False},
        },
        preprocessors=[AnalysisConfig(top_k=2, emit_markdown=False, emit_json=False)],
    )
    processor = AnalysisProcessor(qcfg=qcfg, tokenizer=None)
    layer = _TinyLayer()

    processor.analyze_model(
        layers=[layer],
        layer_modules=[["clean", "outlier"]],
        layers_prefix="model.layers",
    )

    assert [record["name"] for record in processor.records] == ["clean"]
    assert processor.records[0]["bits"] == 8
    assert processor.records[0]["group_size"] == -1
    assert processor.records[0]["sym"] is False
