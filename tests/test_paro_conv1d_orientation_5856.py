# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1

"""Paro must present Conv1D weights to the optimizer in ``[out, in]`` (#5856).

``transformers.Conv1D`` stores ``[in, out]``. Passing that matrix to the Paro
optimizer unchanged made the replay fail with
``addmm: mat1 and mat2 shapes cannot be multiplied`` whenever the two axes
differed, and the packed write-back must restore the storage layout because the
AWQ packer transposes a Conv1D itself.
"""

import tempfile

import pytest
import torch
import transformers
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from gptqmodel import BACKEND, CalibrationConfig, GPTQModel
from gptqmodel.looper.named_module import NamedModule
from gptqmodel.looper.paro_processor import ParoProcessor
from gptqmodel.quantization.config import ParoConfig
from gptqmodel.utils.importer import get_auto_backend_kernel_mapping


IN_FEATURES = 32
OUT_FEATURES = 96
LAYER_NAME = "transformer.h.0.attn.c_attn"


def _named(module: torch.nn.Module) -> NamedModule:
    return NamedModule(module, name="c_attn", full_name=LAYER_NAME, layer_index=0)


def _conv1d():
    # ``Conv1D(nf, nx)`` stores ``[nx, nf]``, and GPT-2 builds its projections
    # as ``Conv1D(out_features, in_features)`` -> ``[in, out]``.
    return transformers.pytorch_utils.Conv1D(OUT_FEATURES, IN_FEATURES)


def test_conv1d_is_presented_out_in():
    layer = _conv1d()
    with torch.no_grad():
        layer.weight.copy_(torch.arange(IN_FEATURES * OUT_FEATURES, dtype=torch.float32).reshape(IN_FEATURES, OUT_FEATURES))

    matrix = ParoProcessor._module_weight_matrix(_named(layer))

    assert tuple(matrix.shape) == (OUT_FEATURES, IN_FEATURES)
    torch.testing.assert_close(matrix, layer.weight.detach().t(), rtol=0, atol=0)


def test_storage_matrix_round_trips_conv1d_and_linear():
    layer = _conv1d()
    with torch.no_grad():
        layer.weight.copy_(torch.randn(IN_FEATURES, OUT_FEATURES))
    wrapped = _named(layer)

    canonical = ParoProcessor._module_weight_matrix(wrapped)
    restored = ParoProcessor._storage_weight_matrix(wrapped, canonical)

    assert tuple(restored.shape) == (IN_FEATURES, OUT_FEATURES)
    torch.testing.assert_close(restored, layer.weight.detach(), rtol=0, atol=0)
    # The restored tensor must own its storage: the AWQ packer transposes it
    # and the module keeps it as `.weight`.
    assert restored.is_contiguous()

    linear = torch.nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False)
    linear_wrapped = NamedModule(linear, name="q_proj", full_name="model.layers.0.q_proj", layer_index=0)
    linear_matrix = ParoProcessor._module_weight_matrix(linear_wrapped)
    assert linear_matrix is linear.weight.data or torch.equal(linear_matrix, linear.weight.data)
    assert ParoProcessor._storage_weight_matrix(linear_wrapped, linear_matrix) is linear_matrix


def test_paro_quantizes_a_gpt2_conv1d_projection():
    """End-to-end: the packed Conv1D layer reproduces its source oracle."""
    torch.set_num_threads(1)
    torch.manual_seed(0)
    get_auto_backend_kernel_mapping()

    with tempfile.TemporaryDirectory() as path:
        dtype = torch.float16
        config = GPT2Config(
            vocab_size=256, n_embd=IN_FEATURES, n_inner=64, n_layer=1, n_head=4, n_positions=256
        )
        config._attn_implementation = "eager"
        model = GPT2LMHeadModel(config).to(dtype).eval()
        source = model.get_submodule(LAYER_NAME)
        with torch.no_grad():
            source.weight.zero_()
            # Two weights on the far corners: any transposed optimizer or
            # write-back changes the projection of the one-hot inputs below.
            source.weight[0, OUT_FEATURES - 1] = 7.0
            source.weight[1, OUT_FEATURES - 2] = -7.0
            source.bias.zero_()
            probe = torch.eye(IN_FEATURES, dtype=dtype)[:2]
            expected = source(probe)
        model.save_pretrained(path)
        tokenizer = Tokenizer(
            models.WordLevel({f"w{i}": i for i in range(config.vocab_size)}, unk_token="w0")
        )
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        PreTrainedTokenizerFast(
            tokenizer_object=tokenizer, unk_token="w0", pad_token="w0"
        ).save_pretrained(path)

        qcfg = ParoConfig(
            bits=4,
            group_size=16,
            sym=True,
            krot=2,
            device="cpu",
            offload_to_disk=False,
            opt_train_samples=16,
            opt_validation_samples=4,
            opt_rotation_epochs=0,
            opt_finetune_epochs=0,
            opt_fused_rotation=False,
            opt_quantizer_impl="fast",
        )
        quantized = GPTQModel.load(path, qcfg, dtype=dtype, backend=BACKEND.PARO)
        calibration = CalibrationConfig(
            dataset=[{"input_ids": list(range(3, 3 + IN_FEATURES)), "attention_mask": [1] * IN_FEATURES}],
            min_length=1,
        )
        quantized.quantize(calibration, backend=BACKEND.PARO)
        layer = quantized.model.get_submodule(LAYER_NAME)

        # One INT32 word holds eight INT4 codes along the output axis.
        assert tuple(layer.qweight.shape) == (IN_FEATURES, OUT_FEATURES // 8)
        with torch.inference_mode():
            torch.testing.assert_close(layer(probe), expected, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(32, 96), (96, 32)])
def test_unequal_axes_are_never_ambiguous(shape):
    in_features, out_features = shape
    linear = torch.nn.Linear(in_features, out_features, bias=False, dtype=torch.float32)
    with torch.no_grad():
        linear.weight.copy_(torch.randn(out_features, in_features))
    layer = transformers.pytorch_utils.Conv1D(out_features, in_features)
    with torch.no_grad():
        layer.weight.copy_(linear.weight.t())

    conv_matrix = ParoProcessor._module_weight_matrix(_named(layer))
    linear_matrix = ParoProcessor._module_weight_matrix(
        NamedModule(linear, name="q_proj", full_name="model.layers.0.q_proj", layer_index=0)
    )

    assert tuple(conv_matrix.shape) == (out_features, in_features)
    torch.testing.assert_close(conv_matrix, linear_matrix, rtol=0, atol=0)
