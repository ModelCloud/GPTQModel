# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""`batch_size=1` and `batch_size>1` must produce the same MoE weights.

Padded calibration rows used to reach the routed-expert and shared-expert
Hessians (see `tests/test_moe_padding_rows.py`), so a padded batch produced
expert weights that differed from the `batch_size=1` run by 14-17% relative
while the dense weights were bit-identical. With padding filtered out of the
2-D hooks and zeroed on the MoE block input, the two runs must agree.
"""

import os
from pathlib import Path


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from transformers import PreTrainedTokenizerFast
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM

from gptqmodel import BACKEND, GPTQModel, QuantizeConfig


pytestmark = [pytest.mark.cpu, pytest.mark.slow]


_LONG_TEXT = (
    "tiny moe calibration sample with enough tokens to survive minimum length "
    "filtering and route through several experts while staying deterministic"
)
_SHORT_TEXT = "short padded sample for the batched calibration run"


def _build_fixture(model_dir: Path) -> Qwen3MoeConfig:
    model_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(
        [_LONG_TEXT, _SHORT_TEXT],
        trainer=WordLevelTrainer(special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"]),
    )
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        pad_token="[PAD]",
    ).save_pretrained(model_dir)
    config = Qwen3MoeConfig(
        num_hidden_layers=1,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=4,
        num_experts_per_tok=2,
        vocab_size=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        max_position_embeddings=128,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    Qwen3MoeForCausalLM(config).save_pretrained(model_dir)
    return config


def _quantize(
    model_dir: Path,
    calibration,
    batch_size: int,
    output: Path,
    patch_model=None,
) -> dict:
    quantize_config = QuantizeConfig(bits=4, group_size=32, device="cpu")
    # FP32 calibration keeps the forward independent of the batch shape: in
    # BF16 the GEMM blocking changes with `batch_size`, which alone moves expert
    # codes by ~10% and would mask the padding effect under test.
    model = GPTQModel.load(
        str(model_dir),
        quantize_config=quantize_config,
        backend=BACKEND.TORCH,
        dtype=torch.float32,
    )
    if patch_model is not None:
        patch_model(model.model)
    model.quantize(
        calibration,
        batch_size=batch_size,
        backend=BACKEND.TORCH,
        calibration_data_min_length=1,
    )
    model.save(output)
    # Compare dequantized weights: packed codes are discrete, so one flipped code
    # is a whole step. Both checkpoints go through the same dequantizer, so the
    # comparison is an A/B of the two calibration runs.
    reloaded = GPTQModel.load(str(output), backend=BACKEND.TORCH, device="cpu")
    weights = {}
    for name, module in reloaded.model.named_modules():
        if not hasattr(module, "dequantize_weight"):
            continue
        if ".mlp.experts." in name or ".self_attn." in name or ".mlp.shared_experts." in name:
            weights[name] = module.dequantize_weight().detach().float().clone()
    return weights


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual = actual.float()
    expected = expected.float()
    return float((actual - expected).norm() / expected.norm().clamp_min(1e-12))


def test_moe_expert_weights_match_across_calibration_batch_sizes(tmp_path: Path):
    model_dir = tmp_path / "native"
    _build_fixture(model_dir)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir)
    # Different lengths force the batched run to pad.
    calibration = [
        dict(tokenizer(_LONG_TEXT, return_tensors="pt")),
        dict(tokenizer(_SHORT_TEXT, return_tensors="pt")),
    ]
    lengths = sorted(int(item["input_ids"].shape[-1]) for item in calibration)
    assert lengths[0] < lengths[1], "the fixture must produce a padded batch"

    single = _quantize(model_dir, calibration, 1, tmp_path / "bs1")
    batched = _quantize(model_dir, calibration, 2, tmp_path / "bs2")

    expert_names = [name for name in single if ".mlp.experts." in name]
    assert expert_names, "the fixture must quantize routed experts"
    for name in expert_names:
        # Pre-fix this was 0.14-0.17 relative; padding is now excluded, so the
        # two runs must agree to floating-point noise.
        assert _relative_error(batched[name], single[name]) < 1e-6, name

    dense_names = [name for name in single if ".self_attn." in name or ".mlp.shared_experts." in name]
    assert dense_names, "the fixture must quantize dense attention"
    for name in dense_names:
        assert _relative_error(batched[name], single[name]) < 1e-6, name


def _force_padding_only_experts(model) -> None:
    """Route every real token to experts 2/3 and every padding row to 0/1.

    The MoE-root hook zeroes padded hidden states before routing. A router sees
    all-zero logits for those rows, so a tie can still send them to experts
    0/1. This router makes that deterministic on every device: experts 0/1 win
    only padding rows, while experts 2/3 win every real token.
    """

    gate = model.model.layers[0].mlp.gate
    num_experts = gate.num_experts
    top_k = gate.top_k
    hidden_dim = gate.hidden_dim

    def forced_forward(hidden_states):
        flat = hidden_states.reshape(-1, hidden_dim)
        real = flat.abs().sum(dim=-1) > 0
        logits = torch.zeros(
            flat.shape[0], num_experts, device=flat.device, dtype=flat.dtype
        )
        logits[~real, 0] = 2.0
        logits[~real, 1] = 1.0
        logits[real, 2] = 2.0
        logits[real, 3] = 1.0
        probs = torch.nn.functional.softmax(logits, dim=-1, dtype=torch.float)
        values, indices = torch.topk(probs, top_k, dim=-1)
        return logits, values.to(logits.dtype), indices

    gate.forward = forced_forward


def test_padding_only_expert_is_not_zeroed_at_batch_size_above_one(tmp_path: Path):
    """An expert that wins no real token must fall back to RTN, not GPTQ.

    Before the fix the padded rows routed to experts 0/1 cleared the 0.5%
    fallback threshold, GPTQ ran on an all-zero Hessian, and
    `_neutralize_dead_columns` zeroed the whole weight.
    """

    model_dir = tmp_path / "native"
    _build_fixture(model_dir)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir)
    calibration = [
        dict(tokenizer(_LONG_TEXT, return_tensors="pt")),
        dict(tokenizer(_SHORT_TEXT, return_tensors="pt")),
        dict(tokenizer(_LONG_TEXT, return_tensors="pt")),
        dict(tokenizer(_SHORT_TEXT, return_tensors="pt")),
    ]

    single = _quantize(
        model_dir, calibration, 1, tmp_path / "bs1", patch_model=_force_padding_only_experts
    )
    batched = _quantize(
        model_dir, calibration, 4, tmp_path / "bs4", patch_model=_force_padding_only_experts
    )

    padding_only = [
        name
        for name in single
        if ".mlp.experts.0." in name or ".mlp.experts.1." in name
    ]
    assert len(padding_only) == 6, padding_only
    for name in padding_only:
        # RTN fallback keeps the source weights; the pre-fix batched run wrote
        # an all-zero tensor here (zero_frac = 1.000).
        assert float(batched[name].abs().max()) > 0.0, name
        assert float((batched[name] == 0).float().mean()) < 1.0, name
        assert _relative_error(batched[name], single[name]) < 1e-6, name
