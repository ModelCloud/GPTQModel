# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Re-saving a loaded quantized fused-expert MoE must keep every expert (#4589).

Transformers 5 builds OLMoE/PhiMoE experts as fused 3-D parameters. The export
shell for a re-save is rebuilt with ``from_config``; if it is not defused before
``make_quant``, the checkpoint's per-expert quantized tensors have no module to
land in and ``save()`` silently writes the fused zero shell. These tests pin the
defuse and the fail-closed guard on a tiny CPU OLMoE model.
"""

import os


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from pathlib import Path  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402
from tokenizers.models import WordLevel  # noqa: E402
from tokenizers.pre_tokenizers import Whitespace  # noqa: E402
from tokenizers.trainers import WordLevelTrainer  # noqa: E402
from transformers import OlmoeConfig, OlmoeForCausalLM, PreTrainedTokenizerFast  # noqa: E402

from gptqmodel import BACKEND, CalibrationConfig, GPTQModel, QuantizeConfig  # noqa: E402
from gptqmodel.quantization.config import ExpertsRoutingOverride, MoEConfig  # noqa: E402


pytestmark = pytest.mark.cpu

_CALIBRATION_TEXTS = [
    "tiny moe calibration sample one with enough tokens to survive minimum length filtering and trigger expert routing",
    "tiny moe calibration sample two with repeated expert words to exercise the moe quantization smoke path cleanly",
    "another synthetic calibration example that is intentionally verbose so token filtering does not remove it",
] * 2

_FUSED_SUFFIXES = ("gate_up_proj", "down_proj")


def _build_olmoe_fixture(model_dir: Path) -> PreTrainedTokenizerFast:
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(
        _CALIBRATION_TEXTS,
        trainer=WordLevelTrainer(special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"]),
    )
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        pad_token="[PAD]",
    )
    fast_tokenizer.save_pretrained(model_dir)

    config = OlmoeConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    OlmoeForCausalLM(config).save_pretrained(model_dir)
    return fast_tokenizer


def _calibration(tokenizer: PreTrainedTokenizerFast):
    dataset = []
    for text in _CALIBRATION_TEXTS:
        encoded = tokenizer(text, return_tensors="pt")
        dataset.append(
            {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}
        )
    return dataset


def _read_safetensors(model_dir: Path):
    with safe_open(model_dir / "model.safetensors", framework="pt") as handle:
        return {key: handle.get_tensor(key) for key in handle.keys()}


@pytest.fixture(scope="module")
def quantized_olmoe(tmp_path_factory):
    root = tmp_path_factory.mktemp("resave4589")
    native = root / "native"
    native.mkdir()
    tokenizer = _build_olmoe_fixture(native)

    quantize_config = QuantizeConfig(
        bits=4,
        group_size=32,
        desc_act=False,
        device="cpu",
        moe=MoEConfig(routing=ExpertsRoutingOverride()),
    )
    model = GPTQModel.load(
        str(native),
        quantize_config=quantize_config,
        backend=BACKEND.TORCH,
    )
    model.quantize(
        CalibrationConfig(
            dataset=_calibration(tokenizer),
            batch_size=1,
            min_length=1,
        ),
        backend=BACKEND.TORCH,
    )
    quantized = root / "quantized"
    model.save(quantized)
    return quantized, root


def test_resave_quantized_olmoe_keeps_every_expert(quantized_olmoe):
    source, root = quantized_olmoe
    resaved = root / "resaved"

    model = GPTQModel.load(str(source), backend=BACKEND.TORCH, device="cpu")
    model.save(resaved)

    original = _read_safetensors(source)
    rewritten = _read_safetensors(resaved)

    assert set(original) == set(rewritten)
    assert not any(
        key.endswith(_FUSED_SUFFIXES) for key in rewritten if ".experts." in key
    )

    expert_keys = [key for key in original if ".experts." in key]
    assert len(expert_keys) == 4 * 3 * 4  # 4 experts x 3 projections x 4 payloads

    nonzero = 0
    for key in expert_keys:
        assert torch.equal(original[key], rewritten[key]), key
        nonzero += original[key].count_nonzero().item()
    assert nonzero > 0

    # The re-saved checkpoint must still load.
    reloaded = GPTQModel.load(str(resaved), backend=BACKEND.TORCH, device="cpu")
    assert reloaded is not None


def test_resave_refuses_when_expert_shell_is_not_defused(quantized_olmoe, monkeypatch):
    source, root = quantized_olmoe
    refused = root / "refused"

    model = GPTQModel.load(str(source), backend=BACKEND.TORCH, device="cpu")

    # Simulate the pre-fix shell: skip the Defuser conversion so the shell keeps
    # its fused 3-D expert parameters and has no slot for the per-expert keys.
    monkeypatch.setattr(
        "gptqmodel.models.loader._convert_model_with_defuser",
        lambda *args, **kwargs: False,
    )

    # Since #4810 the refusal fires earlier, while ``save()`` rebuilds the
    # un-defused shell: the loader rejects the missing per-expert tensors
    # before the re-save guard runs.  Either way nothing is written.
    with pytest.raises(
        (ValueError, RuntimeError), match="Refusing to re-save|does not provide tensors"
    ):
        model.save(refused)

    assert not (refused / "model.safetensors").exists()
