# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""A cancelled or failed ``quantize()`` must restore the in-flight layer (#4546).

The layer stage moves the whole decoder layer (and its non-projection leaves)
to the compute device before a subset forward. Only the success path's
``post_quantize`` moves it back, so a ``StopMainLoop`` from ``subset_callback``
or an exception mid-layer used to leave every parameter and buffer of that
layer on the accelerator. The next ordinary forward then raised
``Expected all tensors to be on the same device`` and the layer's VRAM stayed
pinned until the model was dropped.
"""

from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer, models
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from gptqmodel import BACKEND
from gptqmodel.looper.module_looper import StopMainLoop
from gptqmodel.models.definitions.llama import LlamaQModel
from gptqmodel.quantization import QuantizeConfig
from gptqmodel.quantization.config import AutoDtypeCast


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
CALIBRATION = [{"input_ids": [1, 3, 5, 7, 9, 11, 13, 15], "attention_mask": [1] * 8}]


def _build_qmodel(tmp_path: Path) -> LlamaQModel:
    source = tmp_path / "source"
    config = LlamaConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).eval()
    model.save_pretrained(source)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            models.WordLevel({"[PAD]": 0, "[UNK]": 1, "a": 2}, unk_token="[UNK]")
        ),
        pad_token="[PAD]",
        unk_token="[UNK]",
    )
    return LlamaQModel(
        model=model,
        quantized=False,
        tokenizer=tokenizer,
        quantize_config=QuantizeConfig(
            bits=4,
            group_size=32,
            desc_act=False,
            device=DEVICE,
            offload_to_disk=False,
            auto_dtype_cast=AutoDtypeCast.OFF,
        ),
        model_local_path=str(source),
    )


def _non_cpu_tensors(model) -> list[tuple[str, str]]:
    tensors = [*model.named_parameters(), *model.named_buffers()]
    return [(name, str(tensor.device)) for name, tensor in tensors if tensor.device.type != "cpu"]


def _assert_layer_restored_to_cpu(model) -> None:
    leaked = _non_cpu_tensors(model)
    assert leaked == [], f"tensors left off CPU after a cancelled/failed run: {leaked[:8]}"


def _assert_plain_forward_still_works(model) -> None:
    model.eval()
    with torch.inference_mode():
        output = model(
            input_ids=torch.tensor([[1, 3, 5, 7]]),
            attention_mask=torch.ones(1, 4, dtype=torch.long),
        )
    assert output.logits.shape == (1, 4, 32)


@pytest.mark.parametrize("error", [StopMainLoop, RuntimeError], ids=["cancel", "failure"])
def test_cancelled_or_failed_quantize_restores_inflight_layer(tmp_path, error):
    qmodel = _build_qmodel(tmp_path)

    def cancel(**kwargs):
        # ``forward_start`` runs after the whole layer has been staged on the
        # compute device, which is exactly the mid-layer exit from the report.
        if kwargs["stage"] == "forward_start" and kwargs["layer_idx"] == 0:
            return StopMainLoop
        return None

    def fail(**kwargs):
        # ``quant_start`` runs after the subset's modules moved to the compute
        # device but before the quantizer consumes them.
        if kwargs["stage"] == "quant_start" and kwargs["layer_idx"] == 0:
            raise RuntimeError("injected mid-layer failure")
        return None

    qmodel.subset_callback = cancel if error is StopMainLoop else fail

    with pytest.raises(error):
        qmodel.quantize(
            CALIBRATION,
            batch_size=1,
            backend=BACKEND.GPTQ_TORCH,
            calibration_data_min_length=1,
        )

    _assert_layer_restored_to_cpu(qmodel.model)
    _assert_plain_forward_still_works(qmodel.model)
