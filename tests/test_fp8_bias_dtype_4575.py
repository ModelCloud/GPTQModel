# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""An in-session FP8 module must be the module a checkpoint reload builds.

``TorchFP8Linear.__init__`` allocates the bias as FP32 (the loader keeps FP8
biases in FP32), while the pack path used to keep the model dtype.  That dtype
difference alone selected a different GEMM precision, so a packed module and
the module a reload builds disagreed.
"""

import pytest
import torch

from gptqmodel.nn_modules.qlinear.fp8 import TorchFP8Linear
from tests.fp8_test_utils import pack_fp8_module


_CALIBRATION_TEXTS = [
    "the quick brown fox jumps over the lazy dog and keeps running through the field",
    "a tiny qwen calibration sample with enough tokens to survive length filtering",
    "another synthetic calibration example deliberately verbose so filtering keeps it",
] * 3


def _fp8_module(in_features, out_features, bias):
    return TorchFP8Linear(
        bits=8,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=out_features,
        bias=bias,
        register_buffers=True,
        format="float8_e4m3fn",
    )


def _packed(weight, bias, dtype):
    linear = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=True, dtype=torch.float32)
    with torch.no_grad():
        linear.weight.copy_(weight)
        linear.bias.copy_(bias)
    # A bf16/fp16 run packs a model-dtype bias; the reload path does not.
    linear = linear.to(dtype)
    module = _fp8_module(weight.shape[1], weight.shape[0], bias=True)
    pack_fp8_module(module, linear)
    return module


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fp8_in_session_bias_is_fp32_and_matches_reload(dtype):
    generator = torch.Generator().manual_seed(0)
    weight = torch.randn(32, 64, generator=generator) * 0.02
    bias = torch.randn(32, generator=generator) * 0.05
    in_session = _packed(weight, bias, dtype)

    # A reload builds the module from ``__init__`` (FP32 bias) and copies the
    # checkpoint into it, exactly like the loader does.
    reloaded = _fp8_module(64, 32, bias=True)
    reloaded.load_state_dict(in_session.state_dict())

    assert in_session.bias.dtype == torch.float32
    assert reloaded.bias.dtype == torch.float32
    assert torch.equal(in_session.bias, reloaded.bias)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    in_session = in_session.to(device)
    reloaded = reloaded.to(device)
    x = torch.randn(4, 64, device=device, dtype=dtype)
    with torch.inference_mode():
        actual = in_session(x)
        reloaded_actual = reloaded(x)

    # Identical buffers must take the identical path and produce identical bits.
    assert torch.equal(actual, reloaded_actual)

    # Independent FP64 oracle for the numeric claim.
    decoded = in_session.weight.double() / in_session.weight_scale_inv.double().view(-1, 1)
    oracle = (x.double() @ decoded.T + in_session.bias.double()).to(dtype)
    torch.testing.assert_close(actual, oracle, rtol=2e-3, atol=2e-3)


def _tiny_qwen2_checkpoint(model_dir):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import WordLevelTrainer
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(1234)
    config = Qwen2Config(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
    )
    # Qwen2 attention projections carry a bias, unlike Llama.
    Qwen2ForCausalLM(config).to(torch.bfloat16).eval().save_pretrained(model_dir)

    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(
        _CALIBRATION_TEXTS,
        WordLevelTrainer(special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"]),
    )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        pad_token="[PAD]",
    )
    fast.save_pretrained(model_dir)
    return fast
