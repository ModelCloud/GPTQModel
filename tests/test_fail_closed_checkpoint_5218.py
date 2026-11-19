# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Every quantized kernel must publish its complete checkpoint payload (#5218).

#5032 made the loader fail closed, but only ``qweight`` (GPTQ/AWQ family) and
the QQQ buffers were declared. A checkpoint truncated at ``scales``/``qzeros``
or at any FP8/NF12/GGUF/Paro/VPTQ/BnB/MACaQ/EXL3 tensor loaded into the
zero-filled placeholder: NaN or wrong logits, or a BnB illegal memory access
that poisoned the CUDA context.

The matrix below builds each real kernel module, publishes its ``state_dict()``
as a safetensors checkpoint, and deletes one tensor at a time. Every tensor the
kernel cannot rebuild must raise the fail-closed ``ValueError`` and name the
missing tensor. Tensors the kernel derives (``g_idx`` without activation order,
``code_shape``) or never publishes (``qzeros_format``) are allowed to be absent.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path
from typing import Callable, Tuple

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn


IN_FEATURES = 128
OUT_FEATURES = 64
HOLDER = "layer"


@dataclasses.dataclass(frozen=True)
class _Case:
    build: Callable[[], nn.Module]
    format: object
    required: Tuple[str, ...]
    optional: Tuple[str, ...] = ()


def _torch_linear(**kwargs):
    from gptqmodel.nn_modules.qlinear.torch import TorchLinear
    from gptqmodel.quantization import FORMAT

    return TorchLinear(
        bits=4,
        group_size=32,
        desc_act=kwargs.pop("desc_act", False),
        sym=True,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=False,
        register_buffers=True,
        format=FORMAT.GPTQ_V2,
    )


def _awq_linear(desc_act=False):
    from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
    from gptqmodel.quantization import FORMAT

    return AwqTorchLinear(
        bits=4,
        group_size=32,
        desc_act=desc_act,
        sym=True,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=False,
        register_buffers=True,
        format=FORMAT.GEMM,
    )


def _fp8():
    from gptqmodel.nn_modules.qlinear.fp8 import TorchFP8Linear

    return TorchFP8Linear(
        bits=8,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=False,
        register_buffers=True,
    )


def _nf12():
    from gptqmodel.nn_modules.qlinear.nf12 import NF12Linear

    return NF12Linear(
        bits=12,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=False,
        register_buffers=True,
    )


def _gguf():
    from gptqmodel.nn_modules.qlinear.gguf import GGUFTorchLinear

    return GGUFTorchLinear(
        bits=4,
        group_size=-1,
        sym=True,
        desc_act=False,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=False,
        register_buffers=True,
    )


def _bitsandbytes(bits):
    from gptqmodel.nn_modules.qlinear.bitsandbytes import BitsAndBytesLinear

    return BitsAndBytesLinear(
        bits=bits,
        group_size=64 if bits == 4 else -1,
        sym=True,
        desc_act=False,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=False,
        register_buffers=True,
    )


def _paro():
    from gptqmodel.nn_modules.qlinear.paro import ParoLinear

    return ParoLinear(
        bits=4,
        group_size=32,
        sym=True,
        desc_act=False,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=False,
        register_buffers=True,
    )


def _vptq_plain():
    from gptqmodel.nn_modules.qlinear.vptq_torch import VPTQLinear

    return VPTQLinear(
        bits=4,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=False,
        register_buffers=True,
        num_centroids=(0, 64),
        vector_lens=(0, 8),
    )


def _vptq_full():
    from gptqmodel.nn_modules.qlinear.vptq_torch import VPTQLinear

    return VPTQLinear(
        bits=4,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=False,
        register_buffers=True,
        num_centroids=(64, 64),
        vector_lens=(8, 8),
        num_res_centroids=(64, 64),
        npercent=0.01,
        enable_norm=True,
        enable_perm=True,
    )


def _macaq(rotation="hadamard", pre_mode="balance"):
    from gptqmodel.nn_modules.qlinear.macaq import MacaqLinear

    return MacaqLinear(
        bits=4,
        group_size=-1,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=False,
        register_buffers=True,
        rotation=rotation,
        pre_mode=pre_mode,
    )


def _qqq(group_size, in_features=IN_FEATURES):
    from gptqmodel.nn_modules.qlinear.qqq import QQQTorchLinear

    return QQQTorchLinear(
        bits=4,
        group_size=group_size,
        sym=True,
        desc_act=False,
        in_features=in_features,
        out_features=OUT_FEATURES,
        bias=False,
        register_buffers=True,
    )


def _tt_blockfloat():
    from gptqmodel.nn_modules.qlinear.tt_blockfloat import TTBlockfloatLinear

    return TTBlockfloatLinear(
        bits=4,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=False,
        backend=None,
    )


def _exl3(codebook):
    from gptqmodel.nn_modules.exllamav3 import ExllamaV3Linear

    return ExllamaV3Linear(
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        name=HOLDER,
        tensors={
            "trellis": torch.zeros(8, 4, 64, dtype=torch.int16),
            "suh": torch.zeros(8, dtype=torch.float16),
            "svh": torch.zeros(4, dtype=torch.float16),
            codebook: torch.zeros(1, dtype=torch.int32),
        },
    )


def _exl3_packed():
    from gptqmodel.nn_modules.exllamav3 import ExllamaV3Linear

    return ExllamaV3Linear(
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        name=HOLDER,
        tensors={
            "trellis": torch.zeros(8, 4, 64, dtype=torch.int16),
            "su": torch.zeros(8, dtype=torch.int16),
            "sv": torch.zeros(4, dtype=torch.int16),
            "mcg": torch.zeros(1, dtype=torch.int32),
        },
    )


def _allspark():
    from gptqmodel.nn_modules.qlinear.allspark import AllSparkLinear

    return AllSparkLinear(
        bits=8,
        group_size=-1,
        desc_act=False,
        sym=True,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=False,
        register_buffers=True,
    )


def _bitblas():
    from gptqmodel.nn_modules.qlinear.bitblas import BitblasLinear, bitblas_available

    if not bitblas_available():
        pytest.skip("bitblas is not installed")
    return BitblasLinear(
        bits=4,
        group_size=32,
        desc_act=False,
        sym=True,
        in_features=IN_FEATURES,
        out_features=OUT_FEATURES,
        bias=False,
        dtype=torch.float16,
        register_buffers=True,
    )


def _case(format_name):
    from gptqmodel.quantization import FORMAT

    if format_name == "gptq":
        return _Case(_torch_linear, FORMAT.GPTQ_V2, ("qweight", "scales", "qzeros"), ("g_idx", "qzeros_format"))
    if format_name == "gptq_desc_act":
        return _Case(
            lambda: _torch_linear(desc_act=True),
            FORMAT.GPTQ_V2,
            ("qweight", "scales", "qzeros", "g_idx"),
            ("qzeros_format",),
        )
    if format_name == "awq":
        return _Case(_awq_linear, FORMAT.GEMM, ("qweight", "scales", "qzeros"), ("qzeros_format",))
    if format_name == "fp8":
        return _Case(_fp8, FORMAT.FP8, ("weight", "weight_scale_inv"))
    if format_name == "nf12":
        return _Case(_nf12, FORMAT.NF12, ("nf12_packed", "nf12_escapes", "nf12_exp_base"))
    if format_name == "gguf":
        return _Case(_gguf, FORMAT.GGUF, ("qweight",))
    if format_name == "bnb4":
        return _Case(
            lambda: _bitsandbytes(4),
            FORMAT.BITSANDBYTES,
            (
                "weight",
                "weight_absmax",
                "weight_quant_map",
                "weight_nested_absmax",
                "weight_nested_quant_map",
                "weight_quant_state",
            ),
        )
    if format_name == "bnb8":
        return _Case(lambda: _bitsandbytes(8), FORMAT.BITSANDBYTES, ("weight", "weight_scb"))
    if format_name == "paro":
        return _Case(
            _paro,
            FORMAT.PARO,
            ("qweight", "qzeros", "scales", "theta", "pairs", "channel_scales"),
            ("qzeros_format",),
        )
    if format_name == "vptq_plain":
        return _Case(_vptq_plain, FORMAT.VPTQ, ("indices", "centroids.weight"))
    if format_name == "vptq_full":
        return _Case(
            _vptq_full,
            FORMAT.VPTQ,
            (
                "indices",
                "centroids.weight",
                "res_centroids.weight",
                "outlier_centroids.weight",
                "outlier_indices",
                "weight_scale",
                "weight_bias",
                "perm",
            ),
        )
    if format_name == "macaq":
        return _Case(_macaq, FORMAT.MXFP4_MACAQ, ("W_packed", "W_scales", "BH", "pre_scale"))
    if format_name == "macaq_plain":
        # `rotation="none"`/`pre_mode="none"` never allocate BH/pre_scale; the
        # union declaration must skip them instead of refusing the load.
        return _Case(lambda: _macaq("none", "none"), FORMAT.MXFP4, ("W_packed", "W_scales"))
    if format_name == "qqq":
        return _Case(lambda: _qqq(128, 256), FORMAT.QQQ, ("B", "s_channel", "s_group"))
    if format_name == "qqq_per_channel":
        # `group_size=-1` keeps `s_group` empty; the codec derives it.
        return _Case(lambda: _qqq(-1), FORMAT.QQQ, ("B", "s_channel"), ("s_group",))
    if format_name == "tt_blockfloat":
        return _Case(_tt_blockfloat, FORMAT.TT_BLOCKFLOAT, ("codes", "exponents"), ("code_shape",))
    if format_name == "exl3_mcg":
        return _Case(lambda: _exl3("mcg"), FORMAT.EXL3, ("trellis", "suh", "svh", "mcg"))
    if format_name == "exl3_mul1":
        return _Case(lambda: _exl3("mul1"), FORMAT.EXL3, ("trellis", "suh", "svh", "mul1"))
    if format_name == "exl3_packed":
        return _Case(_exl3_packed, FORMAT.EXL3, ("trellis", "su", "sv", "mcg"))
    if format_name == "allspark":
        # AllSpark has no `qzeros` buffer; the union declaration must skip it.
        return _Case(_allspark, FORMAT.GPTQ_V2, ("qweight", "scales"), ("qzeros_format",))
    if format_name == "bitblas":
        return _Case(_bitblas, FORMAT.GPTQ, ("qweight", "scales"))
    raise AssertionError(format_name)


CASES = (
    "gptq",
    "gptq_desc_act",
    "awq",
    "fp8",
    "nf12",
    "gguf",
    "bnb4",
    "bnb8",
    "paro",
    "vptq_plain",
    "vptq_full",
    "macaq",
    "macaq_plain",
    "qqq",
    "qqq_per_channel",
    "tt_blockfloat",
    "exl3_mcg",
    "exl3_mul1",
    "exl3_packed",
    "allspark",
    "bitblas",
)


class _Holder(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.layer = module


def _save_holder(holder, directory: Path) -> dict[str, torch.Tensor]:
    directory.mkdir(parents=True, exist_ok=True)
    state = {
        f"{HOLDER}.{name}": value.detach().contiguous()
        for name, value in holder.layer.state_dict().items()
    }
    save_file(state, str(directory / "model.safetensors"), metadata={"format": "pt"})
    return state


def _write_without(state, directory: Path, key: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    truncated = {name: value for name, value in state.items() if name != key}
    save_file(truncated, str(directory / "model.safetensors"), metadata={"format": "pt"})


def _build(case_name: str) -> _Case:
    case = _case(case_name)
    if case_name == "allspark" and not torch.cuda.is_available():
        pytest.skip("AllSpark validates a CUDA compute capability at construction")
    try:
        case.build()
    except ImportError as exc:  # optional native dependency
        pytest.skip(str(exc))
    return case


@pytest.mark.parametrize("case_name", CASES)
def test_every_declared_tensor_is_required(tmp_path, case_name):
    from gptqmodel.utils.model import validate_checkpoint_qweights

    case = _build(case_name)
    holder = _Holder(case.build())
    state = _save_holder(holder, tmp_path / "intact")

    # An intact checkpoint passes; this also proves the presence-aware skip
    # accepts the conditional tensors the instance does not own.
    validate_checkpoint_qweights(holder, tmp_path / "intact", case.format)

    # The test's expectation must cover every persistent tensor, so a newly
    # added payload buffer cannot be forgotten here.
    expected = set(case.required) | set(case.optional)
    actual = {name.removeprefix(f"{HOLDER}.") for name in state}
    assert actual == expected, f"{case_name}: unclassified tensors {actual ^ expected}"

    for suffix in case.required:
        key = f"{HOLDER}.{suffix}"
        assert key in state, f"{case_name}: required {suffix} is not published"
        _write_without(state, tmp_path / f"missing-{suffix.replace('.', '_')}", key)
        with pytest.raises(ValueError, match="Missing required quantized weights") as exc:
            validate_checkpoint_qweights(
                holder, tmp_path / f"missing-{suffix.replace('.', '_')}", case.format
            )
        assert key in str(exc.value), f"{case_name}: error did not name {suffix}"

    for suffix in case.optional:
        key = f"{HOLDER}.{suffix}"
        if key not in state:
            continue
        _write_without(state, tmp_path / f"optional-{suffix.replace('.', '_')}", key)
        validate_checkpoint_qweights(
            holder, tmp_path / f"optional-{suffix.replace('.', '_')}", case.format
        )


def _tiny_qwen2(directory: Path) -> None:
    from transformers import Qwen2Config, Qwen2ForCausalLM

    directory.mkdir(parents=True, exist_ok=True)
    config = Qwen2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        tie_word_embeddings=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(5218)
        Qwen2ForCausalLM(config).save_pretrained(directory, safe_serialization=True)


def _quantize_weight_only(source: Path, destination: Path, config) -> None:
    from gptqmodel import GPTQModel

    model = GPTQModel.load(str(source), quantize_config=config)
    model.quantize(backend="torch")
    model.save(str(destination))


def _fp8_checkpoint(source: Path, destination: Path) -> None:
    """Publish a dense checkpoint in a production FP8 layout (row, multiplier).

    FP8 is load-only since #5242, so the fixture cannot come from ``quantize()``
    (#5271). It is encoded with the same reference encoder the FP8 kernel tests
    use, and the checkpoint declares the production scale convention.
    """
    from transformers import AutoModelForCausalLM

    from tests.fp8_test_utils import quantize_fp8_weight

    model = AutoModelForCausalLM.from_pretrained(str(source))
    linear_prefixes = {
        name for name, module in model.named_modules() if isinstance(module, nn.Linear)
    }
    state = {}
    for name, tensor in model.state_dict().items():
        prefix, _, leaf = name.rpartition(".")
        if leaf == "weight" and prefix in linear_prefixes and tensor.dim() == 2:
            quantized, scale_inv = quantize_fp8_weight(tensor, weight_scale_method="row")
            state[f"{prefix}.weight"] = quantized
            state[f"{prefix}.weight_scale_inv"] = scale_inv
        else:
            state[name] = tensor

    shutil.copytree(source, destination)
    save_file(state, str(destination / "model.safetensors"), metadata={"format": "pt"})
    config_path = destination / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["quantization_config"] = {
        "method": "fp8",
        "quant_method": "fp8",
        "format": "float8_e4m3fn",
        "weight_scale_method": "row",
        "weight_scale_semantics": "multiplier",
    }
    config_path.write_text(json.dumps(config), encoding="utf-8")


def _drop(checkpoint: Path, suffix: str) -> None:
    path = checkpoint / "model.safetensors"
    state = load_file(str(path))
    state.pop(f"model.layers.0.self_attn.q_proj.{suffix}")
    save_file(state, str(path), metadata={"format": "pt"})


@pytest.mark.parametrize(
    ("case_name", "build_checkpoint", "required"),
    [
        ("fp8", _fp8_checkpoint, ("weight", "weight_scale_inv")),
        (
            "nf12",
            lambda source, destination: _quantize_weight_only(
                source, destination, _nf12_config()
            ),
            ("nf12_packed", "nf12_escapes", "nf12_exp_base"),
        ),
    ],
)
def test_truncated_weight_only_checkpoint_fails_closed(
    tmp_path, case_name, build_checkpoint, required
):
    """End-to-end: publish, drop one tensor, and load the checkpoint."""

    from gptqmodel import GPTQModel

    source = tmp_path / "source"
    checkpoint = tmp_path / "checkpoint"
    _tiny_qwen2(source)
    build_checkpoint(source, checkpoint)

    model = GPTQModel.load(str(checkpoint), device="cpu")
    with torch.inference_mode():
        logits = model.model(input_ids=torch.tensor([[1, 3, 4, 5]]), use_cache=False).logits
    assert torch.isfinite(logits).all()
    del model

    for suffix in required:
        truncated = tmp_path / f"truncated-{suffix}"
        shutil.copytree(checkpoint, truncated)
        _drop(truncated, suffix)
        with pytest.raises(ValueError, match="Missing required quantized weights") as exc:
            GPTQModel.load(str(truncated), device="cpu")
        assert f"model.layers.0.self_attn.q_proj.{suffix}" in str(exc.value)


def _nf12_config():
    from gptqmodel.quantization.config import NF12Config

    return NF12Config(bits=12)
