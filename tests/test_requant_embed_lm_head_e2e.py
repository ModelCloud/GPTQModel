# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""End-to-end coverage for requantizing both model embedding endpoints."""

from pathlib import Path

import pytest
import torch


_CHECKPOINT = Path("/monster/data/model/Qwen2.5-0.5B-Instruct-gptq-4bit")
_ARC_BATCH_SIZE = 64
_ARC_EXPECTED_BEFORE_REQUANT = {
    "accuracy,loglikelihood": {
        "value": 0.3046075085324232,
        "floor_pct": 0.04,
        "ceil_pct": 0.04,
    },
    "accuracy,loglikelihood_norm": {
        "value": 0.3216723549488055,
        "floor_pct": 0.04,
        "ceil_pct": 0.04,
    },
}
_ARC_EXPECTED_AFTER_REQUANT = {
    "accuracy,loglikelihood": {
        "value": 0.31313993174061433,
        "floor_pct": 0.04,
        "ceil_pct": 0.04,
    },
    "accuracy,loglikelihood_norm": {
        "value": 0.33361774744027306,
        "floor_pct": 0.04,
        "ceil_pct": 0.04,
    },
}

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not _CHECKPOINT.is_dir(), reason="local requant checkpoint is unavailable"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the real-model requant test"),
]


def _evaluate_arc(checkpoint: Path, backend) -> dict[str, float]:
    from tests.eval import evaluate, get_eval_task_metrics

    result = evaluate(
        model_or_id_or_path=str(checkpoint),
        tasks=["arc_challenge"],
        batch_size=_ARC_BATCH_SIZE,
        backend=backend,
        apply_chat_template=False,
        model_args={"device": "cuda:0", "seed": 898},
        gen_kwargs="do_sample=false,temperature=0.0,top_p=1.0,top_k=50",
    )
    return get_eval_task_metrics(result, "arc_challenge")


def _assert_arc_scores(
    actual: dict[str, float],
    expected: dict[str, dict[str, float]],
    *,
    stage: str,
) -> None:
    for metric_name, spec in expected.items():
        assert metric_name in actual, f"ARC metric `{metric_name}` missing {stage}: {actual}"
        expected_value = spec["value"]
        lower = expected_value * (1.0 - spec["floor_pct"])
        upper = expected_value * (1.0 + spec["ceil_pct"])
        assert lower <= actual[metric_name] <= upper, (
            f"ARC metric `{metric_name}` out of range {stage}: "
            f"actual={actual[metric_name]}, expected={expected_value}, range=[{lower}, {upper}]"
        )


def test_requantize_both_save_reload_and_generate(tmp_path):
    """Check ARC before/after requantization, then reload and generate."""

    from gptqmodel import GPTQModel
    from gptqmodel.nn_modules.qlinear import BaseQuantLinear
    from gptqmodel.quantization.config import QuantizeEmbed
    from gptqmodel.utils.backend import BACKEND

    before_arc = _evaluate_arc(_CHECKPOINT, BACKEND.AUTO)
    _assert_arc_scores(before_arc, _ARC_EXPECTED_BEFORE_REQUANT, stage="before requantization")
    torch.cuda.empty_cache()

    model = GPTQModel.load(
        str(_CHECKPOINT),
        device_map={"": "cuda:0"},
        backend=BACKEND.AUTO,
    )
    input_name = model.get_input_embeddings_name()
    output_name = model.get_output_embeddings_name() or model.lm_head
    assert input_name and output_name

    # Keep the endpoint format aligned with the source 4-bit checkpoint while
    # explicitly disabling activation ordering, which has no embedding analogue.
    dynamic = dict(model.quantize_config.dynamic or {})
    endpoint_config = {
        "bits": 4,
        "group_size": 32,
        "sym": True,
        "desc_act": False,
        "act_group_aware": False,
    }
    dynamic[input_name] = dict(endpoint_config)
    dynamic[output_name] = dict(endpoint_config)
    model.quantize_config.dynamic = dynamic

    tokenizer = model.tokenizer
    calibration = tokenizer(
        "The capital of France is Paris. " * 8,
        return_tensors="pt",
        add_special_tokens=True,
    )
    model.requantize(
        calibration=[dict(calibration)],
        embed_quant_mode=QuantizeEmbed.BOTH,
        calibration_concat_size=None,
        calibration_sort=None,
        batch_size=1,
        calibration_data_min_length=8,
        backend=BACKEND.AUTO,
    )

    assert isinstance(model.get_input_embeddings(), BaseQuantLinear)
    assert isinstance(model.get_output_embeddings(), BaseQuantLinear)
    for endpoint_name, endpoint in (
        (input_name, model.get_input_embeddings()),
        (output_name, model.get_output_embeddings()),
    ):
        meta_tensors = [name for name, tensor in endpoint.state_dict().items() if tensor.device.type == "meta"]
        assert not meta_tensors, f"{endpoint_name} retained meta tensors after requantization: {meta_tensors}"

    output_dir = tmp_path / "requantized"
    model.save(str(output_dir))
    del model
    torch.cuda.empty_cache()

    after_arc = _evaluate_arc(output_dir, BACKEND.AUTO)
    _assert_arc_scores(after_arc, _ARC_EXPECTED_AFTER_REQUANT, stage="after requantization")
    torch.cuda.empty_cache()

    reloaded = GPTQModel.load(
        str(output_dir),
        device_map={"": "cuda:0"},
        backend=BACKEND.AUTO,
    )
    assert isinstance(reloaded.get_input_embeddings(), BaseQuantLinear)
    assert isinstance(reloaded.get_output_embeddings(), BaseQuantLinear)

    input_device = next(iter(reloaded.get_input_embeddings().state_dict().values())).device
    prompt = tokenizer("The capital of France is", return_tensors="pt").to(input_device)
    generated = reloaded.generate(prompt, max_new_tokens=4, do_sample=False)
    assert generated.shape[0] == 1
    assert generated.shape[-1] > prompt["input_ids"].shape[-1]
