# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""A requantized embedding must run in-session, on the model device and dtype.

#5116: the requantize loop packs on CPU and a quantized embedding returns the
dtype of its scales (FP16 for GPTQ), so a BF16 model received FP16 embeddings
and the in-session forward failed on a device mismatch before any layer ran.
"""

import tempfile
from pathlib import Path

import pytest
import torch

from gptqmodel import CalibrationConfig, GPTQModel, QuantizeEmbed
from gptqmodel.nn_modules.qlinear.torch import TorchQuantEmbeddings
from gptqmodel.quantization import FORMAT
from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.utils.device import get_device


IN_FEATURES = 64
OUT_FEATURES = 128
GROUP_SIZE = 32


def _packed_embedding(*, force_row_decode: bool = False) -> TorchQuantEmbeddings:
    torch.manual_seed(0)
    module = TorchQuantEmbeddings(
        bits=4,
        group_size=GROUP_SIZE,
        sym=True,
        desc_act=False,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=False,
        format=FORMAT.GPTQ_V2,
    )
    dense = torch.nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False, dtype=torch.float16)
    groups = IN_FEATURES // GROUP_SIZE
    module.pack(
        dense,
        torch.full((OUT_FEATURES, groups), 1 / 128, dtype=torch.float16),
        torch.full((OUT_FEATURES, groups), 1 << 3, dtype=torch.int32),
        torch.arange(IN_FEATURES, dtype=torch.int32) // GROUP_SIZE,
    )
    module.post_init()
    if force_row_decode:
        # The default threshold is 64 MiB; force the row-decode branch.
        module.EMBEDDING_ROW_DECODE_MIN_TABLE_BYTES = 0
    return module.eval()


@pytest.mark.parametrize("force_row_decode", [False, True], ids=["whole_table", "row_decode"])
def test_embedding_forward_returns_the_recorded_output_dtype(force_row_decode):
    """The embedding emits the model dtype, not the checkpoint scale dtype."""

    module = _packed_embedding(force_row_decode=force_row_decode)
    ids = torch.randint(0, IN_FEATURES, (2, 8), dtype=torch.int64)

    with torch.inference_mode():
        default = module(ids)
        module.set_embedding_output_dtype(torch.bfloat16)
        converted = module(ids)
        empty = module(torch.empty((0, 4), dtype=torch.int64))

    assert default.dtype == torch.float16
    assert module.embedding_output_dtype() is torch.bfloat16
    assert converted.dtype == torch.bfloat16
    # Only the visible dtype changes; the dequantized values are the same.
    assert torch.equal(converted, default.to(torch.bfloat16))
    assert empty.dtype == torch.bfloat16
    assert empty.shape == (0, 4, OUT_FEATURES)


def _tiny_llama_dir() -> str:
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    torch.manual_seed(0)
    path = tempfile.mkdtemp(prefix="requant_embed_5116_")
    LlamaForCausalLM(
        LlamaConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=96,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
        )
    ).save_pretrained(path)

    specials = ["<s>", "</s>", "<unk>", "<pad>"]
    tok = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.add_special_tokens(specials)
    tok.train_from_iterator(
        ["the quick brown fox jumps over the lazy dog"] * 20,
        trainer=trainers.WordLevelTrainer(special_tokens=specials, vocab_size=128),
    )
    PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>").save_pretrained(path)
    return path


def _calibration_texts(count: int = 8):
    return [
        " ".join(["the quick brown fox jumps over the lazy dog"] * (1 + index % 3))
        for index in range(count)
    ]


def _quantized_checkpoint() -> str:
    source = _tiny_llama_dir()
    output = str(Path(tempfile.mkdtemp(prefix="requant_embed_5116_q_")) / "quantized")
    model = GPTQModel.load(source, QuantizeConfig(bits=4, group_size=32), device="cpu")
    model.quantize(CalibrationConfig(dataset=_calibration_texts(), batch_size=2))
    model.save(output)
    return output


def _requantize_input(model, *, device):
    dynamic = dict(model.quantize_config.dynamic or {})
    for name in (model.get_input_embeddings_name(), model.get_output_embeddings_name() or model.lm_head):
        dynamic[name] = {"bits": 4, "group_size": 32, "sym": True, "desc_act": False}
    model.quantize_config.dynamic = dynamic
    model.requantize(
        CalibrationConfig(dataset=_calibration_texts(), batch_size=2),
        embed_quant_mode=QuantizeEmbed.INPUT,
    )
    return torch.tensor([[4, 5, 6, 7]], device=device)


def test_requantize_input_embedding_keeps_bf16_and_runs_in_session():
    """A bf16 model keeps a bf16 residual stream and can forward in-session."""

    checkpoint = _quantized_checkpoint()
    model = GPTQModel.load(checkpoint, device="cpu", dtype=torch.bfloat16)
    ids = _requantize_input(model, device="cpu")

    embedding = model.get_input_embeddings()
    assert isinstance(embedding, TorchQuantEmbeddings)
    assert embedding.embedding_output_dtype() is torch.bfloat16

    with torch.inference_mode():
        embedded = embedding(ids)
        logits = model.model(input_ids=ids).logits
    assert embedded.dtype == torch.bfloat16
    assert logits.dtype == torch.bfloat16
    assert torch.isfinite(logits).all()

    saved = str(Path(tempfile.mkdtemp(prefix="requant_embed_5116_r_")) / "requantized")
    model.save(saved)
    reloaded = GPTQModel.load(saved, device="cpu", dtype=torch.bfloat16)
    with torch.inference_mode():
        reloaded_logits = reloaded.model(input_ids=ids).logits
    assert reloaded_logits.dtype == torch.bfloat16
    assert torch.isfinite(reloaded_logits).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the placement check")
def test_requantize_restores_the_endpoint_device():
    """The requantized endpoint runs on the device the loaded model occupies."""

    checkpoint = _quantized_checkpoint()
    model = GPTQModel.load(checkpoint, device_map={"": "cuda:0"}, dtype=torch.bfloat16)
    ids = _requantize_input(model, device="cuda:0")

    embedding = model.get_input_embeddings()
    assert isinstance(embedding, TorchQuantEmbeddings)
    assert get_device(embedding).type == "cuda"
    assert embedding.embedding_output_dtype() is torch.bfloat16

    with torch.inference_mode():
        logits = model.model(input_ids=ids).logits
    assert logits.dtype == torch.bfloat16
    assert torch.isfinite(logits).all()
