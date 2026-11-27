# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# GPU=-1
"""LoRA projections follow the forward-dtype contract (K1/K2).

`contracts/kernel/forward-dtype.md` freezes a module's activation dtype at the
first forward and forbids per-call re-casts kept so a later call in another dtype
still works. `Lora.apply()` used to do exactly that: it re-assigned `lora_A` and
`lora_B` inside every forward whose input dtype differed, which also kept
`optimize()` disabled.

The adapter now aligns once - at the bind site when the module dtype is known,
otherwise on the first eager forward - and raises on a later dtype switch.
"""

import json

import pytest
import torch
from safetensors.torch import save_file
from transformers import LlamaConfig, LlamaForCausalLM

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.adapter.adapter import AdapterCache, Lora
from gptqmodel.quantization import RTNConfig
from gptqmodel.quantization.config import CalibrationConfig
from tests.tiny_remote_models import build_wordlevel_tokenizer


CPU = torch.device("cpu")
MODULE_KEY = "model.layers.0.self_attn.q_proj"


@pytest.fixture(autouse=True)
def _clear_adapter_cache():
    AdapterCache.reset()
    yield
    AdapterCache.reset()


def _write_adapter(directory, *, rank=1, alpha=4, source_a=1.0, source_b=1.0, store=torch.float32):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "adapter_config.json").write_text(
        json.dumps(
            {"r": rank, "lora_alpha": alpha, "use_rslora": False, "target_modules": ["q_proj"]}
        ),
        encoding="utf-8",
    )
    save_file(
        {
            f"base_model.model.{MODULE_KEY}.lora_A.weight": torch.tensor(
                [[source_a]], dtype=store
            ),
            f"base_model.model.{MODULE_KEY}.lora_B.weight": torch.tensor(
                [[source_b]], dtype=store
            ),
        },
        str(directory / "adapter_model.safetensors"),
    )


def test_binding_a_known_dtype_aligns_once_and_locks(tmp_path):
    _write_adapter(tmp_path, source_b=1.0, store=torch.float32)
    adapter = Lora(rank=1, path=str(tmp_path))

    adapter.post_init(MODULE_KEY, CPU, dtype=torch.bfloat16)

    assert adapter.lora_A.dtype == torch.bfloat16
    assert adapter.lora_B.dtype == torch.bfloat16
    assert adapter._forward_dtype == torch.bfloat16


def test_a_dtype_switch_after_binding_raises_instead_of_recasting(tmp_path):
    _write_adapter(tmp_path, source_b=1.0, store=torch.float32)
    adapter = Lora(rank=1, path=str(tmp_path))
    adapter.post_init(MODULE_KEY, CPU, dtype=torch.bfloat16)
    before = (adapter.lora_A, adapter.lora_B)

    with pytest.raises(RuntimeError, match="locked"):
        adapter.apply(
            torch.zeros((1, 1), dtype=torch.float16),
            torch.zeros((1, 1), dtype=torch.float16),
        )

    assert adapter.lora_A is before[0]
    assert adapter.lora_B is before[1]
    assert adapter.lora_A.dtype == torch.bfloat16


def test_same_dtype_forwards_never_touch_the_projections(tmp_path):
    _write_adapter(tmp_path, source_b=1.0, store=torch.float32)
    adapter = Lora(rank=1, path=str(tmp_path))
    adapter.post_init(MODULE_KEY, CPU, dtype=torch.bfloat16)
    x = torch.full((1, 1), 0.125, dtype=torch.bfloat16)
    base = torch.zeros((1, 1), dtype=torch.bfloat16)
    first = (adapter.lora_A, adapter.lora_B)

    adapter.apply(x, base)
    adapter.apply(x, base)

    assert adapter.lora_A is first[0] and adapter.lora_B is first[1]


def test_without_a_dtype_hint_the_first_forward_aligns_once_then_locks(tmp_path):
    _write_adapter(tmp_path, source_b=1.0, store=torch.float32)
    adapter = Lora(rank=1, path=str(tmp_path))
    adapter.post_init(MODULE_KEY, CPU)

    assert adapter._forward_dtype is None

    adapter.apply(
        torch.zeros((1, 1), dtype=torch.float16),
        torch.zeros((1, 1), dtype=torch.float16),
    )
    assert adapter.lora_A.dtype == torch.float16
    assert adapter._forward_dtype == torch.float16

    with pytest.raises(RuntimeError, match="locked"):
        adapter.apply(
            torch.zeros((1, 1), dtype=torch.float32),
            torch.zeros((1, 1), dtype=torch.float32),
        )


@pytest.mark.parametrize("bind_dtype", [torch.bfloat16, None])
def test_the_fold_is_judged_in_the_dtype_the_module_runs_in(tmp_path, bind_dtype):
    """``B = 32768`` with ``alpha / rank = 4`` cannot be folded in FP16 but can in BF16.

    Folding it into FP16 storage turns 131072 into ``inf`` (#5426). Judged in
    the module's real dtype the fold is safe, so the multiply disappears from
    the forward.
    """

    _write_adapter(tmp_path, source_b=32768.0, store=torch.float32)
    adapter = Lora(rank=1, path=str(tmp_path))
    adapter.post_init(MODULE_KEY, CPU, dtype=bind_dtype)

    x = torch.full((1, 1), 0.125, dtype=torch.bfloat16)
    base = torch.zeros((1, 1), dtype=torch.bfloat16)
    expected = float(x.double() @ torch.ones((1, 1), dtype=torch.float64) @ torch.tensor([[32768.0]], dtype=torch.float64) * 4.0)

    if bind_dtype is torch.bfloat16:
        assert adapter._lora_scale_folded is True
        assert torch.isfinite(adapter.lora_B).all()
    else:
        # No dtype known yet: the conservative FP16 bound keeps the product
        # multiply instead of risking an overflow on a later FP16 activation.
        assert adapter._lora_scale_folded is False

    result = adapter.apply(x, base)
    assert torch.isfinite(result).all()
    assert result.item() == pytest.approx(expected, rel=1e-3)


def test_a_float64_adapter_without_a_hint_does_not_lock_to_float32(tmp_path):
    """Storage narrowing is not the activation dtype (review on #6019).

    A float64 export is stored as float32, but a BF16 model must still be able
    to run the first forward: the lock comes from the hint, not from the
    storage rule.
    """

    _write_adapter(tmp_path, source_b=1.0, store=torch.float64)
    adapter = Lora(rank=1, path=str(tmp_path))
    adapter.post_init(MODULE_KEY, CPU)

    assert adapter._forward_dtype is None
    assert adapter.lora_A.dtype == torch.float32

    adapter.apply(
        torch.zeros((1, 1), dtype=torch.bfloat16),
        torch.zeros((1, 1), dtype=torch.bfloat16),
    )

    assert adapter._forward_dtype == torch.bfloat16
    assert adapter.lora_A.dtype == torch.bfloat16

    with pytest.raises(RuntimeError, match="locked"):
        adapter.apply(
            torch.zeros((1, 1), dtype=torch.float16),
            torch.zeros((1, 1), dtype=torch.float16),
        )


def test_a_manual_recast_of_the_projections_is_caught(tmp_path):
    """The lock compares the projections too, not only the activation."""

    _write_adapter(tmp_path, source_b=1.0, store=torch.float32)
    adapter = Lora(rank=1, path=str(tmp_path))
    adapter.post_init(MODULE_KEY, CPU, dtype=torch.bfloat16)
    adapter.lora_A = adapter.lora_A.half()
    adapter.lora_B = adapter.lora_B.half()

    for activation in (torch.float16, torch.bfloat16):
        with pytest.raises(RuntimeError, match="locked"):
            adapter.apply(
                torch.zeros((1, 1), dtype=activation),
                torch.zeros((1, 1), dtype=activation),
            )


def test_another_threads_compile_session_does_not_block_the_lock(tmp_path, monkeypatch):
    """`dynamo_is_compiling()` is per-thread; the raw global call leaked (#5427)."""

    from gptqmodel.adapter import adapter as adapter_module

    _write_adapter(tmp_path, source_b=1.0, store=torch.float32)
    adapter = Lora(rank=1, path=str(tmp_path))
    adapter.post_init(MODULE_KEY, CPU)
    monkeypatch.setattr(adapter_module, "dynamo_is_compiling", lambda: True)

    # Same dtype: the forward runs, but the lock must not be recorded while
    # tracing, exactly like ForwardDtypeLockMixin.
    adapter.apply(
        torch.zeros((1, 1), dtype=torch.float32),
        torch.zeros((1, 1), dtype=torch.float32),
    )
    assert adapter._forward_dtype is None

    # Different dtype while tracing: refuse instead of mutating the projections.
    projected = adapter.lora_A
    with pytest.raises(RuntimeError, match="while tracing"):
        adapter.apply(
            torch.zeros((1, 1), dtype=torch.float16),
            torch.zeros((1, 1), dtype=torch.float16),
        )
    assert adapter.lora_A is projected


def test_align_forward_dtype_locks_the_projections_before_the_first_forward(tmp_path):
    _write_adapter(tmp_path, source_b=1.0, store=torch.float32)
    adapter = Lora(rank=1, path=str(tmp_path))
    adapter.post_init(MODULE_KEY, CPU)

    adapter.align_forward_dtype(torch.bfloat16)

    assert adapter.lora_A.dtype == torch.bfloat16
    assert adapter.lora_B.dtype == torch.bfloat16
    assert adapter._forward_dtype == torch.bfloat16
    # Repeating the same dtype is a no-op, but a different dtype is still a
    # post-forward dtype change.
    adapter.align_forward_dtype(torch.bfloat16)
    with pytest.raises(RuntimeError, match="locked"):
        adapter.align_forward_dtype(torch.float16)


_TINY_VOCAB = 256
_TINY_HIDDEN = 64
_TINY_INTERMEDIATE = 128
_TINY_LAYERS = 1
_TINY_ADAPTER_RANK = 8


def _tiny_llama_config() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=_TINY_VOCAB,
        hidden_size=_TINY_HIDDEN,
        intermediate_size=_TINY_INTERMEDIATE,
        num_hidden_layers=_TINY_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )


def _write_fp32_lora(directory, *, layers, hidden, rank):
    directory.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(6109)
    weights = {}
    for layer in range(layers):
        key = f"model.layers.{layer}.self_attn.q_proj"
        weights[f"base_model.model.{key}.lora_A.weight"] = (
            torch.randn(rank, hidden, dtype=torch.float32) * 0.01
        )
        weights[f"base_model.model.{key}.lora_B.weight"] = (
            torch.randn(hidden, rank, dtype=torch.float32) * 0.01
        )
    save_file(weights, str(directory / "adapter_model.safetensors"))
    (directory / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": rank,
                "lora_alpha": 16,
                "use_rslora": False,
                "target_modules": ["q_proj"],
            }
        ),
        encoding="utf-8",
    )


@pytest.fixture(scope="module")
def fp32_adapter_gptq_checkpoint(tmp_path_factory):
    """A tiny BF16 GPTQ checkpoint with an FP32 LoRA adapter beside it."""

    root = tmp_path_factory.mktemp("adapter-6109")
    base = root / "base"
    base.mkdir()
    torch.manual_seed(6109)
    LlamaForCausalLM(_tiny_llama_config()).to(torch.bfloat16).save_pretrained(base)
    tokenizer = build_wordlevel_tokenizer(vocab_size=_TINY_VOCAB)
    tokenizer.save_pretrained(base)

    texts = [
        "the quick brown fox jumps over the lazy dog and keeps running through the field",
        "a tiny calibration sample with enough tokens to survive length filtering",
        "another synthetic calibration example deliberately verbose for the tokenizer",
        "gravity attracts objects with mass and measurements help test predictions",
    ]
    calibration = [dict(tokenizer(text, return_tensors="pt")) for text in texts]

    model = GPTQModel.load(
        str(base),
        quantize_config=RTNConfig(
            bits=4, group_size=128, device="cpu", offload_to_disk=False
        ),
        backend=BACKEND.TORCH,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model.quantize(
        CalibrationConfig(dataset=calibration, batch_size=1, min_length=1),
        backend=BACKEND.TORCH,
    )
    checkpoint = root / "rtn-4bit-g128"
    model.save(str(checkpoint))
    del model

    adapter = root / "adapter-fp32"
    _write_fp32_lora(
        adapter,
        layers=_TINY_LAYERS,
        hidden=_TINY_HIDDEN,
        rank=_TINY_ADAPTER_RANK,
    )
    return str(checkpoint), str(adapter)


def _load_fp32_adapter_model(checkpoint, adapter):
    return GPTQModel.load(
        checkpoint,
        adapter=Lora(path=adapter, rank=_TINY_ADAPTER_RANK),
        backend=BACKEND.TORCH,
        device="cpu",
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )


def test_optimize_aligns_an_fp32_adapter_before_tracing(fp32_adapter_gptq_checkpoint):
    """``optimize()`` must align the adapter before the traced first forward.

    A quantized linear binds its adapter without a dtype, so the adapter used to
    wait for the first forward to align. ``optimize()`` traces that forward, and
    the alignment raised ``cannot align ... while tracing`` (#6109). The model
    dtype is known before compiling, so ``optimize()`` now aligns every adapter
    first; the traced graph only consumes already-matching projections.
    """

    checkpoint, adapter_path = fp32_adapter_gptq_checkpoint
    model = _load_fp32_adapter_model(checkpoint, adapter_path)
    try:
        q_proj = model.model.model.layers[0].self_attn.q_proj
        assert q_proj.adapter.lora_A.dtype == torch.float32
        assert q_proj.adapter._forward_dtype is None

        # ``eager`` still traces the first forward through dynamo, which is the
        # path that raised, but skips the slow inductor compile.
        model.optimize(backend="eager")
        if not getattr(model, "compiled", False):
            pytest.skip("torch.compile is unavailable on this build")

        ids = model.tokenizer("the quick brown fox", return_tensors="pt")
        out = model.generate(**ids, max_new_tokens=4, do_sample=False)
        assert out.shape[-1] > ids["input_ids"].shape[-1]

        for module in model.model.modules():
            adapter = getattr(module, "adapter", None)
            if not isinstance(adapter, Lora):
                continue
            assert adapter.lora_A.dtype == torch.bfloat16
            assert adapter.lora_B.dtype == torch.bfloat16
            assert adapter._forward_dtype == torch.bfloat16
    finally:
        del model


def test_eager_generate_still_applies_an_fp32_adapter(fp32_adapter_gptq_checkpoint):
    """Eager control: the same fixture works without ``optimize()``."""

    checkpoint, adapter_path = fp32_adapter_gptq_checkpoint
    model = _load_fp32_adapter_model(checkpoint, adapter_path)
    try:
        ids = model.tokenizer("the quick brown fox", return_tensors="pt")
        out = model.generate(**ids, max_new_tokens=4, do_sample=False)
        assert out.shape[-1] > ids["input_ids"].shape[-1]
    finally:
        del model
