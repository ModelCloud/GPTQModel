# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Issues #4863/#5105: a load onto ``cuda:N`` must not initialize ``cuda:0``.

The inference load path hands the device map to Accelerate, which passes the
map value straight to ``safetensors.safe_open(device=...)``. A *string* device
(``"cuda:1"``) makes safetensors initialize the primary context of the current
CUDA device as well, so every non-zero-ordinal load claimed ~500 MiB on a GPU
the caller never asked for. An int ordinal only touches the requested device.

A fused-expert MoE checkpoint (OLMoE) additionally runs Defuser's conversion
before any weight reaches the target GPU. That conversion flushes the cache
through ``clear_memory()``, which synchronizes the *current* device when it has
no device list, so it too used to claim ``cuda:0`` (#5105).

The check runs in a subprocess: a primary context is process-wide and cannot be
released, so an in-process assertion would depend on what ran before it.

Issue #5101 is the same class on the compiled path: ``optimize()`` installs a
compiled ``forward`` that runs without a device guard, and inductor's lazy SDPA
pattern init resolves its device-less default to the *current* device. A model
loaded on ``cuda:1`` therefore claimed ``cuda:0`` when a later generate
recompiled for a new batch shape.
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


CHILD_COMPILED = r"""
import torch
from gptqmodel import GPTQModel
from gptqmodel.utils.torch import dynamo_traced_frames

checkpoint = {checkpoint!r}
model = GPTQModel.load(checkpoint, device="cuda:1", dtype=torch.float16)
print("after_load_ctx0=", torch._C._cuda_hasPrimaryContext(0), flush=True)

model.optimize()
print("compiled=", bool(model.compiled), flush=True)

for batch in (2, 1):
    ids = torch.randint(5, 128, (batch, 8), device="cuda:1")
    out = model.generate(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        max_new_tokens=4,
    )
    print(f"after_batch{{batch}}_ctx0=", torch._C._cuda_hasPrimaryContext(0), flush=True)
    print(f"after_batch{{batch}}_new_tokens=", int(out.shape[1]) - int(ids.shape[1]), flush=True)

print("traced_frames=", dynamo_traced_frames(), flush=True)
"""


_GPTQ_CHECKPOINT_ENV = "GPTQMODEL_DEVICE_CONTEXT_GPTQ_CHECKPOINT"
_GPTQ_CACHE_CANDIDATES = ("sliuau/llama3.2-1b-4bit-group128",)


def _quantized_checkpoint(target: Path, family: str) -> Path:
    """A tiny GPTQ checkpoint, saved the way a user's checkpoint is written."""

    import transformers

    from gptqmodel import GPTQModel
    from gptqmodel.quantization.config import GPTQConfig
    from tests.tiny_remote_models import build_wordlevel_tokenizer

    source = target / "source"
    torch.manual_seed(0)
    if family == "olmoe":
        transformers.OlmoeForCausalLM(
            transformers.OlmoeConfig(
                vocab_size=128,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                num_experts_per_tok=2,
                num_local_experts=4,
                max_position_embeddings=64,
                bos_token_id=1,
                eos_token_id=2,
                pad_token_id=0,
            )
        ).to(torch.float16).save_pretrained(source)
    else:
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


def _visible_cuda_pair() -> str:
    """The first two GPUs this test process may use, as a visibility string."""

    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return "0,1"
    entries = [entry.strip() for entry in raw.split(",") if entry.strip()]
    return ",".join(entries[:2]) if len(entries) >= 2 else "0,1"


@pytest.fixture(scope="module")
def dense_checkpoint(tmp_path_factory) -> Path:
    return _quantized_checkpoint(tmp_path_factory.mktemp("dense"), "dense")


@pytest.fixture(scope="module")
def fused_expert_moe_checkpoint(tmp_path_factory) -> Path:
    return _quantized_checkpoint(tmp_path_factory.mktemp("olmoe"), "olmoe")


@pytest.fixture(scope="module")
def compiled_gptq_checkpoint() -> str:
    """A real 4-bit GPTQ checkpoint whose Marlin graph breaks hit #5101.

    The tiny fixtures above compile without claiming ``cuda:0`` in this
    environment, so they cannot pin the recompile path. Point the test at a
    checkpoint with ``GPTQMODEL_DEVICE_CONTEXT_GPTQ_CHECKPOINT`` or reuse the
    warm cache; skip when neither is available rather than downloading in a
    device-context test.
    """

    configured = os.environ.get(_GPTQ_CHECKPOINT_ENV)
    if configured:
        if not Path(configured).exists():
            pytest.skip(f"{_GPTQ_CHECKPOINT_ENV} points at a missing path: {configured}")
        return configured

    try:
        from huggingface_hub import snapshot_download
    except Exception:  # pragma: no cover - optional test dependency
        pytest.skip("huggingface_hub is unavailable")

    for repo in _GPTQ_CACHE_CANDIDATES:
        try:
            return snapshot_download(repo, local_files_only=True)
        except Exception:
            continue
    pytest.skip(
        "no cached 4-bit GPTQ checkpoint for the compiled device-context "
        f"regression; set {_GPTQ_CHECKPOINT_ENV}"
    )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two visible GPUs")
@pytest.mark.skipif(
    not hasattr(torch._C, "_cuda_hasPrimaryContext"),
    reason="torch does not expose primary-context introspection",
)
@pytest.mark.parametrize("placement", ["cuda:1", {"": "cuda:1"}])
@pytest.mark.parametrize(
    "checkpoint_fixture", ["dense_checkpoint", "fused_expert_moe_checkpoint"]
)
def test_load_on_the_second_gpu_leaves_the_first_untouched(
    request, checkpoint_fixture, placement
):
    checkpoint = request.getfixturevalue(checkpoint_fixture)

    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = _visible_cuda_pair()
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


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two visible GPUs")
@pytest.mark.skipif(
    not hasattr(torch._C, "_cuda_hasPrimaryContext"),
    reason="torch does not expose primary-context introspection",
)
def test_optimize_compiled_forward_on_the_second_gpu_leaves_the_first_untouched(
    compiled_gptq_checkpoint,
):
    """Two compiled generates with different batch sizes must not claim cuda:0.

    The second generate recompiles the forward for a new batch shape, which is
    when inductor's device-less SDPA pattern init used to fall back to the
    current device (cuda:0) instead of the model's device (cuda:1).
    """

    env = os.environ.copy()
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = _visible_cuda_pair()
    env["HF_HUB_OFFLINE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    result = subprocess.run(
        [sys.executable, "-c", CHILD_COMPILED.format(checkpoint=str(compiled_gptq_checkpoint))],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout[-4000:]}\nstderr:\n{result.stderr[-4000:]}"

    assert "after_load_ctx0= False" in result.stdout, result.stdout
    assert "compiled= True" in result.stdout, (
        "optimize() did not install the compiled forward proxy\n"
        f"stdout:\n{result.stdout}"
    )
    assert "after_batch2_ctx0= False" in result.stdout, (
        "the first compiled generate on cuda:1 initialized cuda:0\n"
        f"stdout:\n{result.stdout}"
    )
    assert "after_batch1_ctx0= False" in result.stdout, (
        "a recompile for a new batch shape on cuda:1 initialized cuda:0\n"
        f"stdout:\n{result.stdout}"
    )
    assert "after_batch2_new_tokens= 4" in result.stdout, result.stdout
    assert "after_batch1_new_tokens= 4" in result.stdout, result.stdout

    traced = [
        line.split("=", 1)[1].strip()
        for line in result.stdout.splitlines()
        if line.startswith("traced_frames=")
    ]
    assert traced and int(traced[-1]) > 0, (
        "the compiled path traced no frames; the regression did not exercise it\n"
        f"stdout:\n{result.stdout}"
    )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two visible GPUs")
def test_model_execution_device_scope_restores_the_caller_device():
    """The compiled-call scope enters the model device and always restores."""

    from gptqmodel.utils.torch import model_execution_device_scope

    module = torch.nn.Linear(2, 2, bias=False, device="cuda:1")
    torch.cuda.set_device(0)
    assert torch.cuda.current_device() == 0

    with model_execution_device_scope(module):
        assert torch.cuda.current_device() == 1
    assert torch.cuda.current_device() == 0

    with pytest.raises(RuntimeError, match="compiled call failed"):
        with model_execution_device_scope(module):
            assert torch.cuda.current_device() == 1
            raise RuntimeError("compiled call failed")
    assert torch.cuda.current_device() == 0

    del module


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two visible GPUs")
def test_model_execution_device_prefers_the_first_accelerator_input():
    """A multi-device map follows the call's first accelerator input."""

    from gptqmodel.utils.torch import model_execution_device, model_execution_device_scope

    module = torch.nn.Linear(2, 2, bias=False, device="cuda:1")
    input_ids = torch.zeros((1, 2), dtype=torch.long, device="cuda:1")

    assert model_execution_device(module, (input_ids,)) == torch.device("cuda:1")
    # A CPU-only input falls back to the module's own placement.
    assert model_execution_device(module, (torch.zeros((1, 2)),)) == torch.device("cuda:1")

    torch.cuda.set_device(0)
    with model_execution_device_scope(module, (input_ids,), {}):
        assert torch.cuda.current_device() == 1
    assert torch.cuda.current_device() == 0

    del module, input_ids
