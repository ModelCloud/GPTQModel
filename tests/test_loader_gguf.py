# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from __future__ import annotations

import numpy as np
import torch
from accelerate import init_empty_weights
from transformers import AutoModelForCausalLM, LlamaConfig

from gptqmodel.models.loader import _load_quantized_gguf_checkpoint_into_model
from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear
from gptqmodel.utils import internal_gguf
from gptqmodel.utils.backend import BACKEND
from gptqmodel.utils.model import materialize_meta_tensors


def _tiny_llama():
    return LlamaConfig(
        vocab_size=100,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )


def test_load_quantized_gguf_checkpoint_into_model_fills_meta_slots(monkeypatch):
    """The native quantized GGUF loader should create real tensors directly on the
    target device without requiring a pre-allocation of the full quantized model."""
    config = _tiny_llama()

    with init_empty_weights(include_buffers=True):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=False)
        model.gguf_linear = GGUFTorchLinear(
            bits=4,
            group_size=128,
            sym=True,
            desc_act=False,
            in_features=8,
            out_features=8,
            bias=False,
            pack_dtype=torch.int32,
            backend=BACKEND.GGUF_TORCH,
        )

    # Only computed buffers should be materialized before the GGUF load; quant
    # parameters and persistent weights stay on meta.
    materialize_meta_tensors(model, {"": "cpu"}, only_non_persistent_buffers=True)

    assert model.model.rotary_emb.inv_freq.device.type == "cpu"
    assert model.gguf_linear.qweight.device.type == "meta"
    assert model.lm_head.weight.device.type == "meta"

    qweight_shape = tuple(model.gguf_linear.qweight.shape)
    lm_shape = tuple(model.lm_head.weight.shape)

    ReaderTensor = internal_gguf.ReaderTensor
    fake_tensors = [
        ReaderTensor(
            name="blk.0.weight",
            tensor_type=internal_gguf.GGMLQuantizationType.Q4_0,
            shape=np.array(qweight_shape, dtype=np.uint32),
            n_elements=np.prod(qweight_shape),
            n_bytes=np.prod(qweight_shape),
            data_offset=0,
            data=np.zeros(qweight_shape, dtype=np.uint8),
            field=None,
        ),
        ReaderTensor(
            name="output.weight",
            tensor_type=internal_gguf.GGMLQuantizationType.F16,
            shape=np.array(lm_shape, dtype=np.uint32),
            n_elements=np.prod(lm_shape),
            n_bytes=np.prod(lm_shape) * 2,
            data_offset=0,
            data=np.zeros(lm_shape, dtype=np.float32),
            field=None,
        ),
    ]

    class FakeGGUFReader:
        def __init__(self, *_args, **_kwargs):
            self.tensors = fake_tensors

    def fake_dequantize_to_torch(data, qtype, *, device=None, dtype=torch.float32):
        # data is np.ndarray; return a real tensor on the requested device so we
        # can verify the loader places it correctly without a real GGUF file.
        arr = np.asarray(data)
        if dtype == torch.uint8:
            return torch.from_numpy(arr.astype(np.uint8)).to(device)
        return torch.from_numpy(arr.astype(np.float32)).to(device).to(dtype)

    monkeypatch.setattr("gptqmodel.models.loader.internal_gguf.GGUFReader", FakeGGUFReader)
    monkeypatch.setattr("gptqmodel.models.loader.internal_gguf.dequantize_to_torch", fake_dequantize_to_torch)

    tensor_key_mapping = {
        "blk.0.weight": "gguf_linear.weight",
        "output.weight": "lm_head.weight",
    }

    _load_quantized_gguf_checkpoint_into_model(
        model=model,
        gguf_checkpoint_path="/tmp/fake.gguf",
        tensor_key_mapping=tensor_key_mapping,
        device_map={"": "cpu"},
    )

    # The GGUF loader should have filled the meta slots on the target device.
    assert model.gguf_linear.qweight.device.type == "cpu"
    assert not model.gguf_linear.qweight.is_meta
    assert tuple(model.gguf_linear.qweight.shape) == qweight_shape
    assert model.lm_head.weight.device.type == "cpu"
    assert not model.lm_head.weight.is_meta
    assert tuple(model.lm_head.weight.shape) == lm_shape
