# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Issue #4863: a load onto ``cuda:N`` must not initialize a context on ``cuda:0``.

The inference load path hands the device map to Accelerate, which passes the
map value straight to ``safetensors.safe_open(device=...)``. A *string* device
(``"cuda:1"``) makes safetensors initialize the primary context of the current
CUDA device as well, so every non-zero-ordinal load claimed ~500 MiB on a GPU
the caller never asked for. An int ordinal only touches the requested device.

The check runs in a subprocess: a primary context is process-wide and cannot be
released, so an in-process assertion would depend on what ran before it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]

CHILD = r"""
import torch
from gptqmodel import GPTQModel

checkpoint = {checkpoint!r}
placement = {placement!r}
if isinstance(placement, dict):
    model = GPTQModel.load(checkpoint, device_map=placement, dtype=torch.float16)
else:
    model = GPTQModel.load(checkpoint, device=placement, dtype=torch.float16)

device = next(model.model.parameters()).device
ids = torch.randint(5, 128, (1, 8), device=device)
with torch.inference_mode():
    logits = model.model(input_ids=ids).logits
print("ctx0=", torch._C._cuda_hasPrimaryContext(0), flush=True)
print("ctx1=", torch._C._cuda_hasPrimaryContext(1), flush=True)
print("device=", device, flush=True)
print("finite=", bool(torch.isfinite(logits).all()), flush=True)
"""


def _quantized_checkpoint(target: Path) -> Path:
    """A tiny GPTQ checkpoint, saved the way a user's checkpoint is written."""

    import transformers

    from gptqmodel import GPTQModel
    from gptqmodel.quantization.config import GPTQConfig
    from tests.tiny_remote_models import build_wordlevel_tokenizer

    source = target / "source"
    torch.manual_seed(0)
    transformers.Qwen2ForCausalLM(
        transformers.Qwen2Config(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            tie_word_embeddings=True,
            pad_token_id=0,
        )
    ).to(torch.float16).save_pretrained(source)
    build_wordlevel_tokenizer().save_pretrained(source)

    model = GPTQModel.load(
        str(source),
        quantize_config=GPTQConfig(bits=4, group_size=32, device="cpu"),
        dtype=torch.float16,
    )
    rows = []
    for _ in range(8):
        input_ids = torch.randint(5, 128, (1, 16))
        rows.append({"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)})
    from gptqmodel.quantization.config import CalibrationConfig

    model.quantize(CalibrationConfig(dataset=rows, batch_size=2, min_length=1))
    saved = target / "quantized"
    model.save(str(saved))
    return saved


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two visible GPUs")
@pytest.mark.skipif(
    not hasattr(torch._C, "_cuda_hasPrimaryContext"),
    reason="torch does not expose primary-context introspection",
)
@pytest.mark.parametrize("placement", ["cuda:1", {"": "cuda:1"}])
def test_load_on_the_second_gpu_leaves_the_first_untouched(tmp_path, placement):
    checkpoint = _quantized_checkpoint(tmp_path)

    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = "0,1"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    result = subprocess.run(
        [sys.executable, "-c", CHILD.format(checkpoint=str(checkpoint), placement=placement)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout[-4000:]}\nstderr:\n{result.stderr[-4000:]}"

    assert "ctx1= True" in result.stdout
    assert "finite= True" in result.stdout
    assert "ctx0= False" in result.stdout, (
        "loading onto cuda:1 initialized a primary context on cuda:0\n"
        f"stdout:\n{result.stdout}"
    )
