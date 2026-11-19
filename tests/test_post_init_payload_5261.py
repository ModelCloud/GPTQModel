# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Issue #5261: post-init must probe the kernel payload, not `qweight`.

#5247 routed every ``BaseQuantLinear`` through ``_post_init_offloaded``, which
started with ``module.qweight.is_meta``. NF12, QQQ, VPTQ and BitsAndBytes
store their payload under other names (``nf12_packed``, ``B``/``s_channel``,
``indices``/codebooks, ``weight``), so **every** plain single-GPU load of such
a checkpoint raised::

    AttributeError: 'NF12Linear' object has no attribute 'qweight'

The probe is now a kernel contract: ``checkpoint_payload_tensors()`` resolves
the family's ``CHECKPOINT_REQUIRED_BUFFERS`` declaration (the names the
checkpoint validator already requires) and ``checkpoint_payload_is_meta()``
reports whether Accelerate replaced them with placeholders. A resident layer
runs ``post_init`` in place; a disk-offloaded layer is materialized through
its offload hook, initialized, and returned to meta exactly as #5247
intended.

The load matrix covers one checkpoint per family. FP8 and BitsAndBytes are
load-only, so their payloads are written by hand the same way the existing
FP8/BnB checkpoint tests build them.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import torch
from torch import nn


CALIBRATION_TEXTS = [
    "the quick brown fox jumps over the lazy dog and keeps running through the field",
    "a tiny calibration sample with enough tokens to survive length filtering",
    "another synthetic calibration example deliberately verbose for the tokenizer",
    "gravity attracts objects with mass and measurements help test predictions",
    "the baker mixed flour with water and yeast before leaving the dough to rise",
    "a function transforms inputs into results and tests compare expected values",
    "water evaporates from lakes, clouds form when vapor cools, and rain returns it",
    "a young child watched a small bird spread its wings and fly back to its nest",
]

LINEAR_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("the load matrix needs an accelerator")


def _wordlevel_tokenizer(directory: Path) -> None:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.trainers import WordLevelTrainer
    from transformers import PreTrainedTokenizerFast

    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(
        CALIBRATION_TEXTS,
        WordLevelTrainer(special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"]),
    )
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        pad_token="[PAD]",
    ).save_pretrained(directory)


def _tiny_qwen2_base(root: Path) -> Path:
    from transformers import Qwen2Config, Qwen2ForCausalLM

    base = root / "base"
    if (base / "config.json").exists():
        return base
    base.mkdir(parents=True)
    torch.manual_seed(5261)
    config = Qwen2Config(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        tie_word_embeddings=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    Qwen2ForCausalLM(config).to(torch.float16).eval().save_pretrained(base)
    _wordlevel_tokenizer(base)
    return base


def _calibration(base: Path):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(base))
    return [dict(tokenizer(text, return_tensors="pt")) for text in CALIBRATION_TEXTS]


def _quantize_format(base: Path, destination: Path, name: str) -> None:
    from gptqmodel import BACKEND, CalibrationConfig, GPTQModel
    from gptqmodel.quantization.config import (
        GPTQConfig,
        NF12Config,
        QQQConfig,
        VPTQConfig,
    )

    formats = {
        "gptq": (
            GPTQConfig(bits=4, group_size=128, desc_act=False, device="cuda:0"),
            BACKEND.TORCH,
        ),
        "nf12": (NF12Config(device="cuda:0"), BACKEND.TORCH),
        "qqq": (
            QQQConfig(bits=4, group_size=128, desc_act=False, device="cuda:0"),
            BACKEND.QQQ_TORCH,
        ),
        "vptq": (
            VPTQConfig(
                num_centroids=[-1, 64],
                num_res_centroids=[-1, -1],
                vector_lens=[-1, 8],
                kmeans_iters=8,
                device="cuda:0",
            ),
            BACKEND.VPTQ_TORCH,
        ),
    }
    qcfg, backend = formats[name]
    model = GPTQModel.load(
        str(base),
        quantize_config=qcfg,
        backend=backend,
        dtype=torch.float16,
        attn_implementation="eager",
    )
    model.quantize(
        CalibrationConfig(dataset=_calibration(base), batch_size=1, min_length=1),
        backend=backend,
    )
    model.save(str(destination))


def _replace_quantized_linears(model, factory) -> None:
    for layer in range(model.config.num_hidden_layers):
        for suffix in LINEAR_SUFFIXES:
            name = f"model.layers.{layer}.{suffix}"
            model.set_submodule(name, factory(model.get_submodule(name)))


def _write_fp8_checkpoint(base: Path, destination: Path) -> None:
    from safetensors.torch import save_file
    from transformers import Qwen2ForCausalLM

    from gptqmodel.nn_modules.qlinear.fp8 import TorchFP8Linear
    from gptqmodel.quantization.config import FP8Config
    from tests.fp8_test_utils import pack_fp8_module

    destination.mkdir(parents=True)
    model = Qwen2ForCausalLM.from_pretrained(str(base), dtype=torch.float32).eval()

    def factory(dense):
        module = TorchFP8Linear(
            bits=8,
            group_size=-1,
            sym=True,
            desc_act=False,
            in_features=dense.in_features,
            out_features=dense.out_features,
            bias=dense.bias is not None,
            register_buffers=True,
            format="float8_e4m3fn",
            weight_scale_method="row",
            weight_scale_semantics="inverse",
        )
        pack_fp8_module(module, dense)
        return module

    _replace_quantized_linears(model, factory)
    save_file(
        {name: value.contiguous() for name, value in model.state_dict().items()},
        str(destination / "model.safetensors"),
        metadata={"format": "pt"},
    )
    model.config.quantization_config = FP8Config(
        format="float8_e4m3fn",
        weight_scale_method="row",
        weight_scale_semantics="inverse",
    ).to_dict()
    model.config.save_pretrained(destination)


def _write_bnb_checkpoint(base: Path, destination: Path) -> None:
    from safetensors.torch import save_file
    from transformers import Qwen2ForCausalLM

    from gptqmodel.nn_modules.qlinear.bitsandbytes import BitsAndBytesLinear
    from gptqmodel.quantization.config import BitsAndBytesConfig

    destination.mkdir(parents=True)
    model = Qwen2ForCausalLM.from_pretrained(str(base), dtype=torch.float16).eval()

    def factory(dense):
        module = BitsAndBytesLinear(
            bits=4,
            group_size=-1,
            sym=True,
            desc_act=False,
            in_features=dense.in_features,
            out_features=dense.out_features,
            bias=dense.bias is not None,
            register_buffers=True,
            format="nf4",
            block_size=64,
            compress_statistics=True,
            dtype=torch.float16,
        )
        module.pack_original(linear=dense, scales=None, zeros=None)
        module.post_init()
        return module

    _replace_quantized_linears(model, factory)
    save_file(
        {name: value.contiguous() for name, value in model.state_dict().items()},
        str(destination / "model.safetensors"),
        metadata={"format": "pt"},
    )
    model.config.quantization_config = BitsAndBytesConfig(
        bits=4, format="nf4", block_size=64, compress_statistics=True
    ).to_dict()
    model.config.save_pretrained(destination)


def _copy_tokenizer(base: Path, destination: Path) -> None:
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    ):
        source = base / name
        if source.exists():
            shutil.copy(source, destination / name)


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory):
    """One tiny checkpoint per kernel family, built once for the matrix."""

    _require_cuda()
    root = tmp_path_factory.mktemp("post-init-5261")
    base = _tiny_qwen2_base(root)
    paths = {}
    for name in ("gptq", "nf12", "qqq", "vptq"):
        destination = root / name
        _quantize_format(base, destination, name)
        paths[name] = destination

    fp8 = root / "fp8"
    _write_fp8_checkpoint(base, fp8)
    paths["fp8"] = fp8

    from gptqmodel.nn_modules.qlinear.bitsandbytes import BITSANDBYTES_AVAILABLE

    if BITSANDBYTES_AVAILABLE:
        bnb = root / "bnb"
        _write_bnb_checkpoint(base, bnb)
        paths["bnb"] = bnb
    for destination in paths.values():
        _copy_tokenizer(base, destination)
    return paths


EXPECTED_KERNELS = {
    "gptq": None,
    "nf12": {"NF12Linear"},
    "qqq": {"QQQLinear", "QQQTorchLinear"},
    "vptq": {"VPTQCudaLinear", "VPTQLinear"},
    "fp8": {"TorchFP8Linear"},
    "bnb": {"BitsAndBytesLinear"},
}


def _quantized_modules(model):
    from gptqmodel.nn_modules.qlinear import BaseQuantLinear

    return [
        module for module in model.model.modules() if isinstance(module, BaseQuantLinear)
    ]


def _assert_post_init_side_effect(name, modules):
    """Each family leaves a trace that only its own ``post_init`` writes."""

    if name == "gptq":
        assert all(hasattr(module, "qweight") for module in modules)
    elif name == "nf12":
        assert all(module._exp_base_value is not None for module in modules)
    elif name == "qqq":
        assert all(module.s_channel.dtype is torch.float32 for module in modules)
    elif name == "vptq":
        assert all(module.optimized for module in modules)
    elif name == "fp8":
        assert all(module._decoded_weight_bound_ready for module in modules)
    else:
        # BitsAndBytes exposes no post-init flag; the forward below is its
        # observable contract.
        assert modules


@pytest.mark.parametrize("name", ["gptq", "nf12", "qqq", "vptq", "fp8", "bnb"])
def test_plain_single_gpu_load_runs_post_init_and_forwards(checkpoints, name):
    """`GPTQModel.load(device="cuda:0")` must not read a kernel's `qweight`."""

    from gptqmodel import GPTQModel

    if name not in checkpoints:
        pytest.skip(f"{name} is not available in this runtime")

    model = GPTQModel.load(
        str(checkpoints[name]), device="cuda:0", dtype=torch.float16
    )
    try:
        modules = _quantized_modules(model)
        assert modules, f"{name}: no quantized module was built"
        expected = EXPECTED_KERNELS[name]
        if expected is not None:
            kernels = {type(module).__name__ for module in modules}
            assert kernels <= expected, f"{name}: unexpected kernels {kernels}"
        _assert_post_init_side_effect(name, modules)

        input_ids = torch.tensor(
            [[1, 5, 7, 9, 11]], dtype=torch.long, device="cuda:0"
        )
        with torch.inference_mode():
            logits = model.model(input_ids).logits.float()
        assert logits.shape == (
            1,
            input_ids.shape[1],
            model.model.config.vocab_size,
        )
        assert torch.isfinite(logits).all(), (
            f"{name}: forward produced non-finite logits"
        )
    finally:
        del model
        torch.cuda.empty_cache()


def _kernel_factories():
    from gptqmodel.nn_modules.qlinear.bitsandbytes import BitsAndBytesLinear
    from gptqmodel.nn_modules.qlinear.fp8 import TorchFP8Linear
    from gptqmodel.nn_modules.qlinear.macaq import MacaqLinear
    from gptqmodel.nn_modules.qlinear.nf12 import NF12Linear
    from gptqmodel.nn_modules.qlinear.qqq import QQQTorchLinear

    return {
        "nf12": lambda: NF12Linear(
            bits=12,
            in_features=128,
            out_features=64,
            bias=True,
            register_buffers=True,
            dtype=torch.float16,
        ),
        "qqq": lambda: QQQTorchLinear(
            bits=4,
            group_size=128,
            desc_act=False,
            sym=True,
            in_features=128,
            out_features=128,
            bias=True,
            register_buffers=True,
            dtype=torch.float16,
        ),
        "fp8": lambda: TorchFP8Linear(
            bits=8,
            group_size=-1,
            sym=True,
            desc_act=False,
            in_features=64,
            out_features=32,
            bias=True,
            register_buffers=True,
            format="float8_e4m3fn",
            weight_scale_method="row",
        ),
        "bnb": lambda: BitsAndBytesLinear(
            bits=4,
            group_size=-1,
            sym=True,
            desc_act=False,
            in_features=64,
            out_features=32,
            bias=True,
            register_buffers=True,
            format="nf4",
            block_size=64,
            compress_statistics=True,
            dtype=torch.float16,
        ),
        "macaq": lambda: MacaqLinear(
            bits=4,
            group_size=32,
            desc_act=False,
            sym=True,
            in_features=128,
            out_features=64,
            bias=True,
            register_buffers=True,
            dtype=torch.float16,
        ),
    }


@pytest.mark.parametrize("kernel", ["nf12", "qqq", "fp8", "bnb", "macaq"])
def test_payload_contract_covers_kernels_without_qweight(kernel):
    """The meta probe follows the family payload, not a GPTQ attribute."""

    if kernel == "bnb":
        pytest.importorskip("bitsandbytes", reason="BitsAndBytesLinear needs the optional bitsandbytes package")
    module = _kernel_factories()[kernel]()
    payload = module.checkpoint_payload_tensors()

    assert payload
    assert all(isinstance(tensor, torch.Tensor) for tensor in payload)
    assert not module.checkpoint_payload_is_meta(), kernel
    if kernel != "fp8":
        # FP8 exposes a `qweight` compatibility property for PEFT; the other
        # families never had one, which is what #5261 tripped over.
        assert not hasattr(module, "qweight"), kernel

    module.to("meta")
    assert module.checkpoint_payload_is_meta(), kernel


class _PayloadOnlyLayer(nn.Module):
    """A layer whose payload is `weight`, with no `qweight` attribute."""

    SUPPORTS_CPU_OFFLOAD = True

    def __init__(self):
        super().__init__()
        self.register_buffer("weight", torch.zeros(4))
        self.calls = 0

    def checkpoint_payload_tensors(self):
        return (self.weight,)

    def checkpoint_payload_is_meta(self):
        return any(tensor.is_meta for tensor in self.checkpoint_payload_tensors())

    def post_init(self):
        assert not self.weight.is_meta
        self.calls += 1


def test_post_init_offloaded_materializes_a_layer_without_qweight():
    """A meta layer without `qweight` must be materialized, not AttributeError."""

    import accelerate

    from gptqmodel.utils.model import _post_init_offloaded

    layer = _PayloadOnlyLayer()
    assert not hasattr(layer, "qweight")
    accelerate.cpu_offload(
        nn.Sequential(layer),
        execution_device=torch.device("cpu"),
        offload_buffers=True,
    )
    assert layer.checkpoint_payload_is_meta(), "the fixture must be offloaded"

    _post_init_offloaded(layer)

    assert layer.calls == 1
    # The offload map stays authoritative; the hook replays on forward.
    assert layer.checkpoint_payload_is_meta(), "post_init must restore meta"


def test_post_init_offloaded_keeps_resident_layers_plain():
    """A resident layer needs no hook and runs `post_init` exactly once."""

    from gptqmodel.utils.model import _post_init_offloaded

    layer = _PayloadOnlyLayer()
    _post_init_offloaded(layer)

    assert layer.calls == 1
    assert not layer.checkpoint_payload_is_meta()


def test_post_init_offloaded_fails_closed_without_the_declaration():
    """A CUDA-only `post_init` on a meta layer still raises (#5247 contract)."""

    import accelerate

    from gptqmodel.utils.model import _post_init_offloaded

    class _Undeclared(_PayloadOnlyLayer):
        SUPPORTS_CPU_OFFLOAD = False

    layer = _Undeclared()
    accelerate.cpu_offload(
        nn.Sequential(layer),
        execution_device=torch.device("cpu"),
        offload_buffers=True,
    )

    with pytest.raises(RuntimeError, match="SUPPORTS_CPU_OFFLOAD"):
        _post_init_offloaded(layer)
    assert layer.calls == 0


def _nf12_layer(seed: int):
    """An NF12 layer with a valid payload, built from a random dense linear."""

    from gptqmodel.nn_modules.qlinear.nf12 import NF12Linear

    torch.manual_seed(seed)
    dense = nn.Linear(128, 64, bias=True, dtype=torch.float16)
    layer = NF12Linear(
        bits=12,
        in_features=128,
        out_features=64,
        bias=True,
        register_buffers=True,
        dtype=torch.float16,
    )
    layer.pack_original(linear=dense)
    return layer


def test_gptqmodel_post_init_materializes_a_meta_nf12_layer(monkeypatch):
    """The generic branch must reach a real kernel that has no `qweight`."""

    import accelerate

    from gptqmodel.nn_modules.qlinear.nf12 import NF12Linear
    from gptqmodel.utils.model import gptqmodel_post_init

    layer = _nf12_layer(seed=5261)
    assert not hasattr(layer, "qweight")
    accelerate.cpu_offload(
        nn.Sequential(layer),
        execution_device=torch.device("cpu"),
        offload_buffers=True,
    )
    assert layer.checkpoint_payload_is_meta()

    # NF12's `post_init` only reads the payload, so it is safe to run while
    # materialized. Declare that so the generic branch is exercised instead of
    # the fail-closed CUDA-only error.
    monkeypatch.setattr(NF12Linear, "SUPPORTS_CPU_OFFLOAD", True)

    gptqmodel_post_init(nn.Sequential(layer), use_act_order=False)

    assert layer._exp_base_value is not None, "NF12 post_init did not run"
    assert layer.checkpoint_payload_is_meta(), "the layer must return to meta"


def test_gptqmodel_post_init_runs_post_init_once_per_resident_layer(monkeypatch):
    """A resident layer is initialized exactly once, never double-initialized."""

    from gptqmodel.nn_modules.qlinear.nf12 import NF12Linear
    from gptqmodel.utils.model import gptqmodel_post_init

    layers = [_nf12_layer(seed=1), _nf12_layer(seed=2)]
    calls = []
    original = NF12Linear.post_init

    def counted(self):
        calls.append(id(self))
        return original(self)

    monkeypatch.setattr(NF12Linear, "post_init", counted)

    gptqmodel_post_init(nn.Sequential(*layers), use_act_order=False)

    assert len(calls) == len(layers)
    assert len(set(calls)) == len(layers)
    assert all(layer._exp_base_value is not None for layer in layers)
