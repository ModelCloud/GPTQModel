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

from gptqmodel.adapter.adapter import AdapterCache, Lora


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
