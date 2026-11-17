# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Regression for #4872: AWQ must be able to replay MPT's ``MptMLP`` block.

``MptBlock`` hands the residual stream to the MLP, so
``MptMLP.forward(hidden_states, residual)`` has a required second argument that
the per-linear activation capture never records. The ``norm_2`` scaling group
inspects the ``ffn`` block, so the replay raised
``TypeError: MptMLP.forward() missing 1 required positional argument: 'residual'``
and the whole ``quantize()`` aborted. A ``TypeError`` is not an
``_AWQScaleGroupError``, so the configured fallback never ran either.
"""

import torch
import torch.nn as nn

from gptqmodel.models.definitions.mpt import MptQModel


class _ResidualMLP(nn.Module):
    """Stands in for ``transformers``' ``MptMLP``."""

    def forward(self, hidden_states, residual):
        raise AssertionError("this test only inspects the replay kwargs")


class _PlainBlock(nn.Module):
    def forward(self, hidden_states):
        raise AssertionError("this test only inspects the replay kwargs")


def _wrapper():
    return MptQModel.__new__(MptQModel)


def test_mpt_replay_supplies_a_zero_residual():
    x = torch.randn(6, 16)
    kwargs = {"attention_mask": torch.ones(6, dtype=torch.long)}

    new_x, new_kwargs = MptQModel.awq_module_replay_inputs(_wrapper(), _ResidualMLP(), x, kwargs)

    assert new_x is x
    assert torch.equal(new_kwargs["residual"], torch.zeros_like(x))
    # The captured kwargs must survive untouched.
    assert new_kwargs["attention_mask"] is kwargs["attention_mask"]
    assert "residual" not in kwargs


def test_mpt_replay_keeps_a_captured_residual():
    x = torch.randn(6, 16)
    residual = torch.randn(6, 16)

    _, new_kwargs = MptQModel.awq_module_replay_inputs(
        _wrapper(), _ResidualMLP(), x, {"residual": residual}
    )

    assert new_kwargs["residual"] is residual


def test_mpt_replay_leaves_modules_without_a_residual_alone():
    x = torch.randn(6, 16)
    kwargs = {"attention_mask": torch.ones(6, dtype=torch.long)}

    new_x, new_kwargs = MptQModel.awq_module_replay_inputs(_wrapper(), _PlainBlock(), x, kwargs)

    assert new_x is x
    assert new_kwargs == kwargs


def test_awq_quantize_on_mpt_applies_the_ffn_scale_group(tmp_path):
    """End to end: the ``norm_2 -> ffn`` group must survive the replay."""

    import transformers
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import WordLevelTrainer
    from transformers import PreTrainedTokenizerFast

    from gptqmodel import GPTQModel
    from gptqmodel.quantization.config import FORMAT, AWQConfig, CalibrationConfig

    config = getattr(transformers, "MptConfig", None)
    if config is None:
        import pytest

        pytest.skip("transformers does not provide MptConfig")

    model_dir = tmp_path / "mpt"
    torch.manual_seed(0)
    transformers.MptForCausalLM(
        config(d_model=64, n_heads=4, n_layers=2, vocab_size=128, max_seq_len=128)
    ).to(torch.float16).save_pretrained(model_dir)

    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(
        ["the quick brown fox jumps over the lazy dog"] * 32,
        trainer=WordLevelTrainer(special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"]),
    )
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        pad_token="[PAD]",
    ).save_pretrained(model_dir)

    model = GPTQModel.load(
        str(model_dir),
        quantize_config=AWQConfig(bits=4, group_size=32, format=FORMAT.GEMM, device="cpu"),
        dtype=torch.float16,
    )

    generator = torch.Generator().manual_seed(0)
    rows = []
    for _ in range(16):
        input_ids = torch.randint(5, 128, (1, 16), generator=generator)
        rows.append({"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)})

    model.quantize(CalibrationConfig(dataset=rows, batch_size=4, min_length=1))

    logged = {(int(row["layer"]), str(row["module"])) for row in model.quant_log}
    for layer_index in range(2):
        for projection in ("ffn.up_proj", "ffn.down_proj"):
            assert (layer_index, projection) in logged

    # The quantized model must still produce finite logits.
    with torch.inference_mode():
        logits = model.model(input_ids=torch.randint(5, 128, (1, 16))).logits
    assert torch.isfinite(logits).all()
