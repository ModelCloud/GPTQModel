# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""AWQ clip search must handle a partial output-channel tile (issue #4474).

``AwqTorchLinear`` only requires ``out_features % 8 == 0``, so a projection can
have, for example, 96 output rows. Clip search tiled the output axis at 64/256
rows and asserted divisibility, aborting supported ``quantize()`` runs with a
bare ``AssertionError`` after scale folding and before packing. These tests pin
the padded-tail handling against an independent Torch oracle and run the
reported repro end-to-end through packing and save/reload, with an exact-tile
control shape.
"""

import gc
import os
from pathlib import Path


# The reported repro is CPU-only; keep model construction off any visible GPU.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from gptqmodel import BACKEND, CalibrationConfig, GPTQModel, QuantizeConfig
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.quantization import FORMAT, METHOD
from tests.test_awq_clip_consistency import _ClipTestAWQProcessor, _sequential_clip_grid
from tests.test_tiny_moe_quant_smoke import _build_calibration_dataset, _build_local_tokenizer


pytestmark = [pytest.mark.cpu, pytest.mark.slow]


@pytest.mark.parametrize(
    "out_features",
    [8, 96, 128],
    ids=["sub-tile", "partial-64-plus-32", "exact-tile"],
)
def test_clip_bounds_cover_every_output_row(out_features):
    """The 96-row tail must be clipped exactly like the independent per-row oracle."""

    generator = torch.Generator().manual_seed(4474)
    weight = torch.randn(out_features, 96, generator=generator)
    input_feat = torch.randn(13, 96, generator=generator)
    qcfg = QuantizeConfig(
        quant_method=METHOD.AWQ, format=FORMAT.GEMM, group_size=32,
        mse={"clip_grid_budget": 0},
    )
    processor = _ClipTestAWQProcessor(qcfg)

    actual = processor._compute_best_clip(weight, input_feat)
    expected = _sequential_clip_grid(qcfg, weight, input_feat)

    # One bound per (output row, group); the trailing singleton is the group axis
    # `apply_clip` reshapes against.
    assert actual.shape == (out_features, 3, 1)
    assert torch.isfinite(actual).all()
    assert (actual >= 0).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_clip_search_rejects_non_2d_weight():
    """A genuinely invalid shape keeps an explicit error instead of a bare assert."""

    qcfg = QuantizeConfig(quant_method=METHOD.AWQ, format=FORMAT.GEMM, group_size=32)
    processor = _ClipTestAWQProcessor(qcfg)

    with pytest.raises(ValueError, match="2-D weight tensor"):
        processor._compute_best_clip(torch.ones(2, 3, 4), torch.ones(5, 4))


@pytest.mark.parametrize(
    "hidden_size",
    [96, 128],
    ids=["partial-output-tile", "exact-output-tile"],
)
def test_awq_quantize_partial_output_tile_cpu(tmp_path: Path, hidden_size: int):
    """The issue's 96-row repro must quantize, pack, save and reload on CPU."""

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(31)
        model_dir = tmp_path / "native"
        config = Qwen3Config(
            num_hidden_layers=1,
            hidden_size=hidden_size,
            intermediate_size=2 * hidden_size,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=hidden_size // 4,
            vocab_size=128,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
        Qwen3ForCausalLM(config).save_pretrained(model_dir)
        tokenizer = _build_local_tokenizer(model_dir)
        calibration = _build_calibration_dataset(tokenizer)

        quantize_config = QuantizeConfig(
            bits=4, group_size=32, desc_act=False, device="cpu",
            method=METHOD.AWQ, format=FORMAT.GEMM, offload_to_disk=False,
        )
        model = GPTQModel.load(str(model_dir), quantize_config=quantize_config, backend=BACKEND.TORCH)
        model.quantize(
            CalibrationConfig(
                dataset=calibration,
                batch_size=1,
                min_length=1,
            ),
            backend=BACKEND.TORCH,
        )

        packed = {
            name: module for name, module in model.model.named_modules()
            if isinstance(module, AwqTorchLinear)
        }
        assert packed, "AWQ quantize() did not pack any projection"
        widths = sorted({module.out_features for module in packed.values()})
        if hidden_size % 64:
            assert any(width % 64 for width in widths)
        else:
            assert all(width % 64 == 0 for width in widths)
        for module in packed.values():
            # AWQ packs output-channel groups into the trailing axis.
            assert module.qweight.dtype == torch.int32
            assert module.qweight.shape == (module.in_features, module.out_features // 8)

        batch = calibration[0]
        with torch.inference_mode():
            logits = model.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            ).logits
        assert torch.isfinite(logits).all()

        destination = tmp_path / "quantized"
        model.save(str(destination))
        assert list(destination.glob("*.safetensors"))
        del model, packed
        gc.collect()

        restored = GPTQModel.load(
            str(destination), device="cpu", backend=BACKEND.AWQ_TORCH, dtype=torch.float16,
        )
        with torch.inference_mode():
            restored_logits = restored.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            ).logits
        assert torch.isfinite(restored_logits).all()
        del restored
        gc.collect()
    finally:
        torch.set_num_threads(previous_threads)
