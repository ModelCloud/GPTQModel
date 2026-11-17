# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1

"""Regression tests for #4887: AWQ must pack GPT-2 ``Conv1D`` projections.

GPT-2's ``c_attn``/``c_proj``/``c_fc`` are ``transformers.Conv1D`` modules,
which store ``weight`` as ``[in_features, out_features]``. The AWQ processor
and packers treated every source weight as a Linear ``[out, in]`` matrix, so a
non-square Conv1D crashed while packing and a square one (``c_proj``) would
have been packed transposed. These tests cover the packer contract directly
and a tiny end-to-end GPT-2 AWQ round trip on CPU.
"""

import gc

import pytest
import torch
import transformers
from tokenizers import Tokenizer, models, pre_tokenizers
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from gptqmodel import BACKEND, CalibrationConfig, GPTQModel
from gptqmodel.nn_modules.qlinear.gemv_awq import AwqGEMVLinear
from gptqmodel.nn_modules.qlinear.gemv_fast_awq import AwqGEMVFastLinear
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.quantization.config import AWQConfig


_VOCAB_SIZE = 1000
_N_EMBD = 128
_N_HEAD = 4
_N_LAYER = 2


def _tiny_tokenizer() -> PreTrainedTokenizerFast:
    """A local tokenizer so the fixture never downloads a checkpoint."""

    backend = Tokenizer(
        models.WordLevel(
            vocab={f"tok{i}": i for i in range(_VOCAB_SIZE)},
            unk_token="tok0",
        )
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="tok0",
        pad_token="tok0",
        bos_token="tok1",
        eos_token="tok2",
    )


def _tiny_gpt2(tmp_path):
    torch.manual_seed(0)
    model_dir = tmp_path / "tiny-gpt2"
    config = GPT2Config(
        n_embd=_N_EMBD,
        n_head=_N_HEAD,
        n_layer=_N_LAYER,
        n_positions=64,
        vocab_size=_VOCAB_SIZE,
    )
    model = GPT2LMHeadModel(config).to(torch.bfloat16).eval()
    model.save_pretrained(model_dir)
    _tiny_tokenizer().save_pretrained(model_dir)
    return model_dir, model


def _calibration_rows() -> list:
    generator = torch.Generator().manual_seed(4887)
    rows = []
    for _ in range(8):
        ids = torch.randint(3, _VOCAB_SIZE, (64,), generator=generator)
        rows.append({"input_ids": ids.tolist(), "attention_mask": [1] * ids.numel()})
    return rows


def _inputs() -> tuple:
    generator = torch.Generator().manual_seed(4887)
    input_ids = torch.randint(3, _VOCAB_SIZE, (2, 32), generator=generator)
    return input_ids, torch.ones_like(input_ids)


@pytest.mark.cpu
@pytest.mark.integration
def test_awq_gpt2_conv1d_quantize_save_reload(tmp_path):
    model_dir, reference_model = _tiny_gpt2(tmp_path)
    input_ids, attention_mask = _inputs()
    with torch.inference_mode():
        reference = reference_model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False,
        ).logits.float()
    assert torch.isfinite(reference).all()

    model = GPTQModel.load(
        str(model_dir),
        AWQConfig(bits=4, group_size=128, device="cpu", offload_to_disk=False),
        dtype=torch.bfloat16,
        backend=BACKEND.AWQ_TORCH,
    )
    model.quantize(
        CalibrationConfig(dataset=_calibration_rows(), batch_size=1, min_length=1),
        backend=BACKEND.AWQ_TORCH,
    )

    # GPT-2 has four Conv1D projections per block: c_attn and c_proj in
    # attention, then c_fc and c_proj in the MLP.
    packed = [module for module in model.model.modules() if isinstance(module, AwqTorchLinear)]
    assert len(packed) == 4 * _N_LAYER
    for module in packed:
        assert module.qweight.dtype == torch.int32
        assert module.qweight.shape[0] == module.in_features

    with torch.inference_mode():
        actual = model.model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False,
        ).logits.float()
    assert torch.isfinite(actual).all()
    relative_rmse = ((actual - reference).square().mean() / reference.square().mean()).sqrt().item()
    # 4-bit quantization of a random tiny model is noisy; the bound only has to
    # catch a transposed or mis-grouped weight, which collapses the logits.
    print(f"AWQ_CONV1D_METRICS relative_logit_rmse={relative_rmse:.6f}", flush=True)
    assert relative_rmse < 0.30, relative_rmse

    before = {name: tensor.detach().clone() for name, tensor in model.model.state_dict().items()}
    destination = tmp_path / "quantized"
    model.save(str(destination))
    del model, packed
    gc.collect()

    restored = GPTQModel.load(
        str(destination),
        dtype=torch.bfloat16,
        device="cpu",
        backend=BACKEND.AWQ_TORCH,
    )
    after = restored.model.state_dict()
    assert before.keys() == after.keys()
    for name, tensor in before.items():
        assert torch.equal(tensor, after[name].cpu()), name

    with torch.inference_mode():
        reloaded = restored.model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False,
        ).logits.float()
    torch.testing.assert_close(reloaded, actual, rtol=1e-3, atol=1e-3)
    del restored
    gc.collect()


@pytest.mark.cpu
@pytest.mark.parametrize(
    "module_cls",
    [AwqTorchLinear, AwqGEMVLinear, AwqGEMVFastLinear],
    ids=["torch", "gemv", "gemv_fast"],
)
@pytest.mark.parametrize("in_features,out_features", [(128, 384), (128, 128)])
def test_awq_pack_conv1d_matches_linear_layout(module_cls, in_features, out_features):
    """A Conv1D source must pack exactly like its Linear transpose.

    The three packers that can be built without a CUDA kernel are covered
    here. The GEMM/Triton/Marlin/Machete AWQ packers resolve their kernel at
    construction, so their identical ``source_weight_out_in`` call site is
    only exercised on a CUDA host.

    ``128 -> 128`` is the square case (GPT-2 ``attn.c_proj``): there a missing
    transpose does not raise, it silently packs the transposed matrix.
    """

    torch.manual_seed(4887)
    bits, group_size = 4, 128
    groups = in_features // group_size

    linear = nn.Linear(in_features, out_features, bias=True)
    conv1d = transformers.Conv1D(nx=in_features, nf=out_features)
    conv1d.weight.data.copy_(linear.weight.data.t())
    conv1d.bias.data.copy_(linear.bias.data)

    # Derive real affine parameters from the weight itself. Random scales large
    # enough to avoid overflow would saturate every code at 2**bits - 1, and a
    # saturated code matrix cannot expose a transposed pack.
    weight_groups = linear.weight.data.view(out_features, groups, group_size)
    weight_min = weight_groups.min(dim=2).values
    weight_max = weight_groups.max(dim=2).values
    scales = ((weight_max - weight_min) / ((1 << bits) - 1)).to(torch.float16)
    zeros = torch.round(-weight_min / scales.float()).to(torch.int32)
    # Guard the fixture: an out-of-range zero point or a zero scale would
    # saturate the codes and hide a transposed pack.
    assert scales.min() > 0 and torch.isfinite(scales).all()
    assert zeros.min() >= 0 and zeros.max() <= (1 << bits) - 1

    def pack(source):
        module = module_cls(
            bits=bits,
            group_size=group_size,
            sym=True,
            desc_act=False,
            in_features=in_features,
            out_features=out_features,
            bias=True,
            register_buffers=True,
        )
        module.pack(source, scales.clone(), zeros.clone())
        return module

    linear_module = pack(linear)
    conv1d_module = pack(conv1d)
    assert torch.equal(linear_module.qweight, conv1d_module.qweight)
    assert torch.equal(linear_module.qzeros, conv1d_module.qzeros)
    assert torch.equal(linear_module.scales, conv1d_module.scales)
    assert torch.equal(linear_module.bias, conv1d_module.bias)

    if module_cls is AwqTorchLinear:
        # Only the Torch backend can also run a CPU forward.
        linear_module.post_init()
        conv1d_module.post_init()
        x = torch.randn(3, in_features, dtype=torch.float16)
        with torch.inference_mode():
            assert torch.equal(linear_module(x), conv1d_module(x))
