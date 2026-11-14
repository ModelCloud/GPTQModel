# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Save-time materialization must not replace a packed FP8 payload.

Issue #4037: ``TorchFP8Linear`` keeps its packed codes in a buffer named
``weight``, the same leaf name as the source checkpoint tensor. With the
default ``offload_to_disk=True`` the packed buffer is META, and the writer's
``alias_all_from_turtle_if_meta`` pass used to fill it from the dense source
checkpoint before ``get_state_dict_for_save`` could restore the offload bundle.
The saved checkpoint then carried the original weights next to the packed
scale, which reloads as garbage (PPL 1e38).

These tests pin both halves of the contract: the writer skips modules that
declare a ``QUANT_TYPE``, and a real FP8 quantize -> save -> reload round trip
keeps the packed codes and a sane perplexity.
"""

import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

from gptqmodel.nn_modules.qlinear.fp8 import TorchFP8Linear
from gptqmodel.utils.structure import LazyTurtle, alias_all_from_turtle_if_meta


QWEN_MODEL_ID = "/monster/data/model/Qwen2.5-0.5B-Instruct"
QWEN_CALIBRATION = [
    "The capital of France is Paris, and the river that flows through it is the Seine.",
    "Machine learning models learn patterns from data and generalize to new examples.",
    "A quantized linear layer stores packed codes and a per-row or per-block scale.",
    "Reading a book every day improves vocabulary and writing style over time.",
    "The temperature of the ocean rises slowly because water has a high heat capacity.",
    "A reliable test fails before the fix and passes after the fix is applied.",
    "Chess players study openings, middlegames, and endgames to improve their play.",
    "The library closes at eight in the evening on weekdays and at six on Saturday.",
]
QWEN_EVAL_TEXT = (
    "The quick brown fox jumps over the lazy dog. Quantization reduces the memory "
    "footprint of large language models by storing weights in a lower precision "
    "format. A good checkpoint must round trip exactly, because a silently "
    "corrupted checkpoint produces fluent but wrong answers. Perplexity is a "
    "standard measure of how well a language model predicts a held out text. "
    "Calibration data should be representative of the target domain so that the "
    "quantization error stays small. When the saved tensor is replaced by the "
    "original dense weight, the scale no longer matches and the outputs explode. "
) * 4


class _Shell(nn.Module):
    """Minimal shell: one packed projection plus one dense parameter owner."""

    def __init__(self):
        super().__init__()
        self.q_proj = TorchFP8Linear(
            bits=8,
            group_size=-1,
            sym=True,
            desc_act=False,
            in_features=32,
            out_features=32,
            bias=True,
            register_buffers=True,
            format="float8_e4m3fn",
            weight_scale_method="row",
        )
        self.norm = nn.Linear(32, 32, bias=False)


def _write_checkpoint_index(model_dir: Path, shard_name: str, keys) -> None:
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(keys, shard_name)}),
        encoding="utf-8",
    )


def _build_lazy_turtle(tmp_path: Path, tensors: dict[str, torch.Tensor]) -> LazyTurtle:
    model_dir = tmp_path / "source_model"
    model_dir.mkdir()
    shard_name = "model.safetensors"
    save_file(tensors, str(model_dir / shard_name))
    _write_checkpoint_index(model_dir, shard_name, tensors)
    turtle = LazyTurtle.maybe_create(
        model_local_path=str(model_dir),
        config=SimpleNamespace(_experts_implementation=None),
        model_init_kwargs={"device_map": {"": "cpu"}},
    )
    assert turtle is not None
    return turtle


def _meta_shell() -> _Shell:
    shell = _Shell()
    # Model the default disk offload: the packed payload is META and the
    # authoritative bytes live in the offload bundle.
    shell.to_empty(device="meta")
    return shell


def test_quantized_module_types_covers_fp8_payload_owner():
    from gptqmodel.models.writer import _quantized_module_types

    shell = _Shell()
    assert _quantized_module_types(shell) == (TorchFP8Linear,)

    dense = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
    assert _quantized_module_types(dense) == ()


def test_save_materialization_skips_packed_fp8_payload(tmp_path):
    from gptqmodel.models.writer import _quantized_module_types

    source_weight = torch.randn(32, 32) * 1.2
    turtle = _build_lazy_turtle(
        tmp_path,
        {
            "q_proj.weight": source_weight.clone(),
            "q_proj.bias": torch.zeros(32),
            "norm.weight": torch.randn(32, 32),
        },
    )

    fixed = _meta_shell()
    alias_all_from_turtle_if_meta(
        shell_model=fixed,
        turtle_model=turtle,
        skip_module_types=_quantized_module_types(fixed),
    )
    # The packed payload stays META for get_state_dict_for_save() to restore
    # from the offload bundle; the dense neighbour is materialized as usual.
    assert fixed.q_proj.weight.is_meta
    assert fixed.q_proj.weight_scale_inv.is_meta
    assert not fixed.norm.weight.is_meta

    # Control: without the skip the dense source replaces the packed codes.
    control = _meta_shell()
    alias_all_from_turtle_if_meta(shell_model=control, turtle_model=turtle)
    assert not control.q_proj.weight.is_meta
    expected = source_weight.to(torch.float8_e4m3fn)
    assert torch.equal(control.q_proj.weight.view(torch.uint8), expected.view(torch.uint8))


def _read_safetensors_key(directory: str, key: str) -> torch.Tensor:
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".safetensors"):
            continue
        with safe_open(os.path.join(directory, name), framework="pt", device="cpu") as handle:
            if key in handle.keys():
                return handle.get_tensor(key)
    raise KeyError(f"{key} not found under {directory}")


def _held_out_ppl(model, tokenizer, text: str, device) -> float:
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    with torch.inference_mode():
        logits = model(input_ids=ids).logits.float()
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.size(-1)),
        ids[:, 1:].reshape(-1),
        reduction="mean",
    )
    return math.exp(loss.item())


@pytest.mark.model
@pytest.mark.slow
@pytest.mark.skipif(
    not os.path.isdir(QWEN_MODEL_ID),
    reason=f"model fixture {QWEN_MODEL_ID} is not available",
)
def test_fp8_default_offload_save_round_trip_keeps_packed_codes(tmp_path):
    from transformers import AutoTokenizer

    from gptqmodel import GPTQModel
    from gptqmodel.quantization import METHOD
    from gptqmodel.quantization.config import QuantizeConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_ID)
    qcfg = QuantizeConfig(quant_method=METHOD.FP8, format="float8_e4m3fn", bits=8)
    assert qcfg.offload_to_disk is True

    model = GPTQModel.load(QWEN_MODEL_ID, quantize_config=qcfg, device=device)
    model.quantize(QWEN_CALIBRATION, batch_size=1, backend="torch")

    # The offload bundle is written from module.state_dict() at pack time, so it
    # holds the in-memory packed codes that the checkpoint must preserve.
    offload_dir = os.path.join(
        model.quantize_config.offload_to_disk_path,
        "model.layers.0.self_attn.q_proj",
    )
    packed_codes = _read_safetensors_key(offload_dir, "weight")
    assert packed_codes.dtype == torch.float8_e4m3fn
    packed_absmax = float(packed_codes.float().abs().max())
    assert packed_absmax > 100.0

    save_dir = tmp_path / "fp8"
    model.save(str(save_dir))
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    saved = _read_safetensors_key(str(save_dir), "model.layers.0.self_attn.q_proj.weight")
    assert saved.dtype == torch.float8_e4m3fn
    assert torch.equal(saved.view(torch.uint8), packed_codes.view(torch.uint8))

    source = _read_safetensors_key(QWEN_MODEL_ID, "model.layers.0.self_attn.q_proj.weight")
    assert float(source.float().abs().max()) < 10.0

    reloaded = GPTQModel.load(str(save_dir), device=device)
    ppl = _held_out_ppl(reloaded.model, tokenizer, QWEN_EVAL_TEXT, device)
    assert math.isfinite(ppl)
    assert ppl < 50.0
