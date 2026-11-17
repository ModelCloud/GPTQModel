# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""`finalize()` overrides must run the shared cleanup (#4747).

`NativeProcessor.finalize` was the only override that did not chain to
`LoopProcessor.finalize`, so the last layer's captured native inputs stayed
reachable through the `ModuleLooper -> processors` reference cycle and
`quantize()` returned with them still allocated on the GPU until a cyclic GC.
The AWQ packer kept the last `mlp.down_proj`'s dense fake-quantized weight the
same way.

These tests are deterministic and CPU-only: they assert the cleanup contract
rather than measuring `torch.cuda.memory_allocated()`, which on a tiny fixture
is too small to distinguish.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from gptqmodel import BACKEND, CalibrationConfig, GPTQModel
from gptqmodel.looper.awq_processor import AWQProcessor
from gptqmodel.looper.native_processor import NativeProcessor
from gptqmodel.quantization.config import AWQConfig, GPTAQConfig, QuantizeConfig


LOOPER_DIR = Path(__file__).resolve().parents[1] / "gptqmodel" / "looper"


def _finalize_overrides():
    """Every `def finalize` in a looper module, as (module, class, node)."""

    for path in sorted(LOOPER_DIR.glob("*.py")):
        if path.name in ("loop_processor.py", "input_cache.py"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "finalize":
                    yield path.name, node.name, item


def _calls_super_finalize(node: ast.FunctionDef) -> bool:
    for call in ast.walk(node):
        if not isinstance(call, ast.Call):
            continue
        func = call.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "finalize"
            and isinstance(func.value, ast.Call)
            and isinstance(func.value.func, ast.Name)
            and func.value.func.id == "super"
        ):
            return True
    return False


def test_every_finalize_override_chains_to_the_base():
    overrides = list(_finalize_overrides())
    assert overrides, "no finalize overrides found; the AST probe is broken"
    missing = [f"{module}:{cls}" for module, cls, node in overrides if not _calls_super_finalize(node)]
    assert not missing, f"finalize overrides that skip LoopProcessor.finalize: {missing}"


def test_native_finalize_releases_the_shared_cleanup():
    processor = NativeProcessor(
        tokenizer=None,
        qcfg=QuantizeConfig(bits=4, group_size=8, gptaq=GPTAQConfig()),
        calibration=CalibrationConfig(dataset=None, batch_size=1),
    )
    processor.native_inp_caches = {"model.layers.0.self_attn.q_proj": [torch.zeros(1, 8)]}
    processor.inputs_cache = {"sentinel": torch.zeros(1, 8)}
    processor._results = {"sentinel": object()}

    processor.finalize(model=SimpleNamespace(quantize_config=None))

    assert not processor.native_inp_caches, "the native capture cache must be dropped"
    assert not hasattr(processor, "inputs_cache"), "LoopProcessor.finalize must drop inputs_cache"
    assert not hasattr(processor, "_results"), "LoopProcessor.finalize must drop _results"


def _tiny_model(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            tie_word_embeddings=False,
        )
    ).save_pretrained(directory)
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="[BOS]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        pad_token="[PAD]",
    ).save_pretrained(directory)


def _calibration():
    return [{"input_ids": torch.randint(0, 64, (1, 16))} for _ in range(2)]


@pytest.mark.slow
def test_gptaq_quantize_clears_the_native_caches_before_returning(tmp_path, monkeypatch):
    """The NativeProcessor that ran the loop must be cleaned before return."""

    _tiny_model(tmp_path / "source")
    seen = []
    original = NativeProcessor.finalize

    def spy(self, model, **kwargs):
        original(self, model=model, **kwargs)
        seen.append(self)

    monkeypatch.setattr(NativeProcessor, "finalize", spy)

    model = GPTQModel.load(
        str(tmp_path / "source"),
        quantize_config=QuantizeConfig(
            bits=4, group_size=32, device="cpu", offload_to_disk=False,
            gptaq=GPTAQConfig(device="cpu"),
        ),
        backend=BACKEND.TORCH,
    )
    model.quantize(CalibrationConfig(dataset=_calibration(), batch_size=1, min_length=1), backend=BACKEND.TORCH)

    assert seen, "the GPTAQ plan never ran a NativeProcessor"
    processor = seen[0]
    assert getattr(processor, "native_inp_caches", {}) == {}, (
        "the last layer's native inputs were still reachable after quantize()"
    )
    assert not hasattr(processor, "inputs_cache"), "LoopProcessor.finalize never ran"
    assert not hasattr(processor, "_results"), "LoopProcessor.finalize never ran"


@pytest.mark.slow
def test_awq_pack_module_drops_the_dense_fake_quantized_weight(tmp_path, monkeypatch):
    """`wq` is only needed for the diff; it must not outlive pack_module()."""

    _tiny_model(tmp_path / "source")
    leftover = []
    original = AWQProcessor.pack_module

    def spy(self, module):
        original(self, module)
        if "wq" in module.state:
            leftover.append(module.full_name)

    monkeypatch.setattr(AWQProcessor, "pack_module", spy)

    model = GPTQModel.load(
        str(tmp_path / "source"),
        quantize_config=AWQConfig(bits=4, group_size=32, device="cpu", offload_to_disk=False),
        backend=BACKEND.TORCH,
    )
    model.quantize(CalibrationConfig(dataset=_calibration(), batch_size=1, min_length=1), backend=BACKEND.TORCH)

    assert not leftover, f"packed modules kept their dense wq: {leftover}"
