# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import nn

from gptqmodel.looper.named_module import NamedModule
from gptqmodel.looper.qvq_processor import QVQProcessor
from gptqmodel.nn_modules.qlinear.qvq import QVQLinear
from gptqmodel.quantization import (
    ModuleGranularReplayConfig,
    QuantizeConfig,
    QVQConfig,
)
from gptqmodel.quantization.qvq import QVQQuantizationTelemetry
from gptqmodel.quantization.qvq_rates import qvq_words_per_tile
from scripts.validate_qvq_p4_live_prefix import _parser


class _TinyDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(16, 16, bias=False)

    def forward(self, hidden):
        return torch.tanh(self.q_proj(hidden))


class _TinyFourLayerCausalLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(32, 16)
        self.layers = nn.ModuleList([_TinyDecoderLayer() for _ in range(4)])
        self.lm_head = nn.Linear(16, 32, bias=False)

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False, use_cache=False, return_dict=True):
        del attention_mask, use_cache, return_dict
        hidden = self.embed_tokens(input_ids)
        hidden_states = [hidden]
        for layer in self.layers:
            hidden = layer(hidden)
            hidden_states.append(hidden)
        return type(
            "TinyOutput",
            (),
            {
                "logits": self.lm_head(hidden),
                "hidden_states": tuple(hidden_states) if output_hidden_states else None,
            },
        )()


def _prepared_calibration(**kwargs):
    return kwargs["calibration_dataset"]


def _replay_rows(*starts):
    return [
        {
            "input_ids": torch.tensor([[start, start + 1, start + 2, start + 3]], dtype=torch.long),
            "attention_mask": torch.ones((1, 4), dtype=torch.long),
        }
        for start in starts
    ]


def _replay_processor(search, confirmation, *, device="cpu", yaqa=None):
    qcfg = _base_config(module_granular_replay=True, device=device)
    processor = QVQProcessor(
        tokenizer=None,
        qcfg=qcfg,
        calibration=_replay_rows(21),
        prepare_dataset_func=_prepared_calibration,
        calibration_concat_size=None,
        calibration_sort=None,
        batch_size=1,
        yaqa_calibration=yaqa,
        module_replay_search_calibration=search,
        module_replay_confirmation_calibration=confirmation,
    )
    return processor, qcfg


def _base_config(**kwargs):
    return QVQConfig(
        bits=2,
        format="qvq_v2b2_p32",
        rounding="yaqa",
        offload_to_disk=False,
        **kwargs,
    )


def test_module_granular_replay_is_default_off_and_boolean_true_uses_safe_defaults():
    baseline = _base_config()
    enabled = _base_config(module_granular_replay=True)

    assert baseline.module_granular_replay is None
    assert isinstance(enabled.module_granular_replay, ModuleGranularReplayConfig)
    assert enabled.module_granular_replay.subsets == ("attention_qkvo",)
    assert enabled.module_granular_replay.roles() == ("q_proj", "v_proj", "k_proj", "o_proj")
    assert enabled.module_granular_replay.alternative_bank_ids == (1, 2, 3)
    assert enabled.module_granular_replay.search_folds == 2
    assert enabled.module_granular_replay.require_disjoint_confirmation is True


def test_module_granular_replay_round_trips_without_changing_inference_layout():
    baseline = _base_config()
    configured = _base_config(
        module_granular_replay={
            "subsets": ["attention_qk", "mlp_gate_up_down"],
            "module_order": [
                "q_proj",
                "k_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            "alternative_bank_ids": [1, 3],
            "search_folds": 3,
            "minimum_relative_kl_improvement": 0.002,
            "topn_regression_limit": 0.001,
        },
    )

    payload = configured.to_dict()
    reloaded = QuantizeConfig.from_quant_config(payload)

    assert reloaded.to_dict() == payload
    assert reloaded.module_granular_replay.roles() == (
        "q_proj",
        "k_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    assert reloaded.module_granular_replay.includes_module("model.layers.2.self_attn.q_proj")
    assert not reloaded.module_granular_replay.includes_module("model.layers.2.self_attn.v_proj")
    assert configured.quant_linear_init_kwargs() == baseline.quant_linear_init_kwargs()


def test_atomic_swiglu_replay_keeps_gate_up_down_in_one_subset():
    processor = QVQProcessor(
        tokenizer=None,
        qcfg=_base_config(
            module_granular_replay={
                "strategy": "atomic_swiglu",
                "subsets": ["mlp_gate_up_down"],
            }
        ),
        calibration=_replay_rows(21),
        prepare_dataset_func=_prepared_calibration,
        calibration_concat_size=None,
        calibration_sort=None,
        batch_size=1,
        module_replay_search_calibration=_replay_rows(1, 5),
        module_replay_confirmation_calibration=_replay_rows(9, 13),
    )

    groups = processor.refine_subset_module_groups(
        [
            ["model.layers.0.mlp.gate_proj", "model.layers.0.mlp.up_proj"],
            ["model.layers.0.mlp.down_proj"],
        ]
    )

    assert groups == [[
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.mlp.up_proj",
        "model.layers.0.mlp.down_proj",
    ]]


def test_atomic_swiglu_strategy_requires_complete_mlp_subset():
    with pytest.raises(ValueError, match="mlp_gate_up_down"):
        _base_config(
            module_granular_replay={
                "strategy": "atomic_swiglu",
                "subsets": ["mlp_gate_up"],
            }
        )


def test_atomic_swiglu_requires_reselect_canonical_family():
    with pytest.raises(ValueError, match="v2b2_family_mode='reselect'"):
        _base_config(
            yaqa={"v2b2_family_mode": "fixed_block_ldlq"},
            module_granular_replay={
                "strategy": "atomic_swiglu",
                "subsets": ["mlp_gate_up_down"],
            },
        )


def test_live_prefix_driver_exposes_module_granular_replay_name_and_subset_scope():
    args = _parser().parse_args(
        [
            "--model",
            "model",
            "--dataset",
            "dataset",
            "--prefix-artifact",
            "prefix.safetensors",
            "--yaqa-factor-cache",
            "factors.pt",
            "--yaqa-metadata",
            "factors.json",
            "--output",
            "report.json",
            "--module-granular-replay",
            "--replay-subsets",
            "attention_vo",
        ]
    )

    assert args.module_granular_replay is True
    assert args.replay_subsets == ["attention_vo"]
    assert args.module_replay_search_folds == 2
    assert args.replay_folds == 1


@pytest.mark.parametrize(
    ("replay", "message"),
    (
        ({"subsets": ["unknown"]}, "unsupported subsets"),
        ({"subsets": ["attention_qk"], "module_order": ["q_proj"]}, "missing.*k_proj"),
        ({"alternative_bank_ids": [0]}, "alternative bank IDs"),
        ({"alternative_bank_ids": [1, 1]}, "must not contain duplicates"),
        ({"strategy": "beam"}, "greedy"),
        ({"replay_horizon": "subset"}, "final_logits"),
        ({"search_folds": 1}, "at least two"),
        ({"require_disjoint_confirmation": False}, "disjoint confirmation"),
        ({"fallback": "best_local"}, "canonical_v2_yaqa"),
    ),
)
def test_module_granular_replay_rejects_unvalidated_controls(replay, message):
    with pytest.raises(ValueError, match=message):
        _base_config(module_granular_replay=replay)


@pytest.mark.parametrize(
    "config_kwargs",
    (
        {"format": "qvq", "rounding": "yaqa"},
        {"format": "qvq_v2b2_p32", "rounding": "block_ldlq"},
        {
            "format": "qvq_v2b2_p32",
            "rounding": "yaqa",
            "yaqa": {"sample_strategy": "64_16x16"},
        },
        {
            "format": "qvq_v2b2_p32",
            "rounding": "yaqa",
            "yaqa": {"spectral_refinement": True},
        },
    ),
)
def test_module_granular_replay_requires_exact_v2b2_yaqa_candidates(config_kwargs):
    with pytest.raises(ValueError, match="module-granular replay|sample_strategy"):
        QVQConfig(
            bits=2,
            module_granular_replay=True,
            offload_to_disk=False,
            **config_kwargs,
        )


def test_module_granular_replay_teacher_cache_uses_disjoint_rows_and_four_layer_logits():
    torch.manual_seed(20260818)
    search = _replay_rows(1, 5)
    confirmation = _replay_rows(9, 13)
    processor, _ = _replay_processor(search, confirmation)
    model = _TinyFourLayerCausalLM().eval()
    wrapper = type("TinyWrapper", (), {"model": model})()

    processor.prepare_module_granular_replay(wrapper)

    assert len(processor._module_replay_teacher_logits["search"]) == 2
    assert len(processor._module_replay_teacher_logits["confirmation"]) == 2
    assert processor._module_replay_teacher_logits["search"][0].shape == (1, 3, 32)
    metrics = processor._module_replay_metrics("search")
    assert metrics["kl_forward"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["top1_agreement"] == 1.0
    assert metrics["top5_overlap"] == 1.0
    assert metrics["top10_overlap"] == 1.0


def test_module_granular_replay_metrics_exclude_padding_and_padding_boundaries():
    search = _replay_rows(1, 5)
    confirmation = _replay_rows(9, 13)
    for row in (*search, *confirmation):
        row["attention_mask"] = torch.tensor([[0, 1, 1, 1]], dtype=torch.long)
    processor, _ = _replay_processor(search, confirmation)
    wrapper = type("TinyWrapper", (), {"model": _TinyFourLayerCausalLM().eval()})()

    processor.prepare_module_granular_replay(wrapper)
    metrics = processor._module_replay_metrics("search")

    # Each four-token row has two valid next-token edges: 1->2 and 2->3.
    assert metrics["tokens"] == 4
    assert metrics["kl_forward"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["top1_agreement"] == 1.0


def test_module_granular_replay_rejects_overlapping_search_and_confirmation_prompts():
    shared = _replay_rows(1, 5)
    processor, _ = _replay_processor(shared, _replay_rows(5, 9))
    wrapper = type("TinyWrapper", (), {"model": _TinyFourLayerCausalLM().eval()})()

    with pytest.raises(ValueError, match="prompt-disjoint"):
        processor.prepare_module_granular_replay(wrapper)


@pytest.mark.parametrize(
    ("search", "yaqa", "message"),
    (
        (_replay_rows(21, 5), None, "ordinary calibration"),
        (_replay_rows(1, 5), _replay_rows(5, 30), "YAQA calibration"),
    ),
)
def test_module_granular_replay_rejects_calibration_prompt_reuse(search, yaqa, message):
    processor, _ = _replay_processor(search, _replay_rows(9, 13), yaqa=yaqa)
    wrapper = type("TinyWrapper", (), {"model": _TinyFourLayerCausalLM().eval()})()

    with pytest.raises(ValueError, match=message):
        processor.prepare_module_granular_replay(wrapper)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_module_granular_replay_keeps_complete_model_on_replay_device():
    processor, _ = _replay_processor(_replay_rows(1, 5), _replay_rows(9, 13), device="mps")
    model = _TinyFourLayerCausalLM().eval()
    wrapper = type("TinyWrapper", (), {"model": model})()

    processor.prepare_module_granular_replay(wrapper)

    assert {tensor.device.type for tensor in model.state_dict().values()} == {"mps"}
    model.layers[0].to("cpu")
    assert {tensor.device.type for tensor in model.state_dict().values()} == {"cpu", "mps"}
    processor._ensure_module_replay_residency(model)
    assert {tensor.device.type for tensor in model.state_dict().values()} == {"mps"}
    assert processor._module_replay_metrics("search")["top1_agreement"] == 1.0


def test_module_granular_replay_candidate_selection_restores_live_dense_module():
    torch.manual_seed(20260818)
    processor, qcfg = _replay_processor(_replay_rows(1, 5), _replay_rows(9, 13))
    model = _TinyFourLayerCausalLM().eval()
    wrapper = type("TinyWrapper", (), {"model": model})()
    processor.prepare_module_granular_replay(wrapper)
    original = model.layers[0].q_proj
    named = NamedModule(
        original,
        name="q_proj",
        full_name="layers.0.q_proj",
        layer_index=0,
    )
    identity = torch.eye(16, dtype=torch.float32)

    result = processor._select_module_granular_replay_candidate(
        named,
        qcfg,
        original.weight.detach().to(torch.float32),
        identity,
        {
            "output_hessian": identity,
            "seed": 7,
            "codebook_version": qcfg.codebook,
            "vector_size": 2,
            "trellis_window": 16,
            "v2b2_p32": True,
            "rounding": "yaqa",
            "bank_count": 2,
            "telemetry": QVQQuantizationTelemetry(),
        },
    )

    assert model.layers[0].q_proj is original
    assert result.bank_ids is not None
    assert result.telemetry is not None
    replay = processor._module_replay_stats["layers.0.q_proj"]
    assert replay["selected_alternative_bank_id"] in (0, 1, 2, 3)
    assert len(replay["candidates"]) == 4
    assert sum(record["selected"] for record in replay["candidates"]) == 1


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_module_replay_qlinear_post_init_owns_versioned_mps_buffers():
    with torch.inference_mode():
        tensors = {
            "trellis": torch.zeros((1, qvq_words_per_tile(2)), dtype=torch.int32, device="mps"),
            "SU": torch.ones(16, dtype=torch.float32, device="mps"),
            "SV": torch.ones(16, dtype=torch.float32, device="mps"),
            "bank_ids": torch.zeros(1, dtype=torch.uint8, device="mps"),
            "bank_alt_id": torch.ones(1, dtype=torch.uint8, device="mps"),
        }
        module = QVQLinear(
            bits=2,
            in_features=16,
            out_features=16,
            bias=False,
            dtype=torch.float16,
            tensors=tensors,
            bank_count=2,
            v2b2_p32=True,
        ).eval()
        module.post_init()

    with torch.inference_mode():
        module.to("cpu")
        module.to("mps")
    module.post_init()

    for name in ("trellis", "SU", "SV", "bank_ids", "bank_alt_id"):
        tensor = getattr(module, name)
        assert not tensor.is_inference()
        assert isinstance(tensor._version, int)
    first_packed = module._qvq_mps_bank_ids
    module.bank_ids.fill_(255)
    second_packed = module._prepare_mps_bank_ids(torch.device("mps"))
    assert second_packed is not first_packed
    assert torch.equal(second_packed.cpu(), torch.full((1,), 255, dtype=torch.uint8))


def test_module_granular_replay_serializes_selected_subset_modules_in_greedy_order():
    processor, _ = _replay_processor(_replay_rows(1, 5), _replay_rows(9, 13))

    groups = processor.refine_subset_module_groups(
        [["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"], ["self_attn.o_proj"]]
    )

    assert groups == [
        ["self_attn.q_proj"],
        ["self_attn.v_proj"],
        ["self_attn.k_proj"],
        ["self_attn.o_proj"],
    ]
