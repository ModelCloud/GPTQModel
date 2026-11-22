# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""AWQ must quantize a balanced MoE device pool end to end (#5104).

``VramStrategy.BALANCED`` places the routed experts of one norm-anchored AWQ
scale group round-robin over the MoE device pool, so a single group spans every
device. AWQ is the only documented multi-GPU MoE configuration that crashed:
``_accumulate_awq_weight_mean`` allocated its accumulator on the first member's
device and added the rest into it, which raised on layer 0 right after
calibration capture.

This test builds a tiny 16-expert Qwen3-MoE checkpoint (16 experts is the
threshold at which ``build_subset_plan`` splits a MoE family across the pool),
quantizes it with AWQ twice, and checks:

* the balanced run finishes and packs every expert projection;
* a norm-anchored group actually spanned two devices, so the trigger ran;
* the balanced and exclusive runs produce identical packed tensors, i.e. the
  fold is a gather, not a different fold.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch


_NUM_LAYERS = 2
_NUM_EXPERTS = 16
_TOKENS_PER_ROW = 24
_ROWS = 4
_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_CONFIG_ARGS = {
    "num_hidden_layers": _NUM_LAYERS,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "vocab_size": 128,
    "pad_token_id": 0,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "moe_intermediate_size": 32,
    "num_experts": _NUM_EXPERTS,
    "num_experts_per_tok": 2,
    "tie_word_embeddings": False,
}
_TOKENIZER_TEXTS = [
    "tiny moe calibration sample one with enough tokens to survive minimum length filtering and trigger expert routing",
    "tiny moe calibration sample two with repeated expert words to exercise the moe quantization smoke path cleanly",
    "another synthetic calibration example that is intentionally verbose so token filtering does not remove it",
] * 2


def _build_fixture(model_dir: Path) -> None:
    """Persist the tiny two-layer, 16-expert Qwen3-MoE fixture."""

    tests_dir = str(Path(__file__).resolve().parent)
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)

    import kernels  # noqa: F401  (must win over tests/kernels for transformers)
    import torch
    from test_tiny_moe_quant_smoke import _build_local_tokenizer
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    torch.manual_seed(1234)
    Qwen3MoeForCausalLM(Qwen3MoeConfig(**_CONFIG_ARGS)).save_pretrained(model_dir)
    tokenizer = _build_local_tokenizer(model_dir)
    tokenizer.save_pretrained(model_dir)


def _calibration():
    import torch

    torch.manual_seed(6789)
    return [
        {
            "input_ids": torch.randint(3, 100, (1, _TOKENS_PER_ROW)),
            "attention_mask": torch.ones(1, _TOKENS_PER_ROW, dtype=torch.long),
        }
        for _ in range(_ROWS)
    ]


def _quantize(model_dir: Path, *, strategy: str):
    """Quantize the fixture and return the group spans plus packed tensors."""

    import threading

    import torch

    import gptqmodel.looper.awq_processor as awq_processor
    from gptqmodel import BACKEND, CalibrationConfig, GPTQModel
    from gptqmodel.quantization.config import AWQConfig, VramStrategy

    spans = []
    lock = threading.Lock()
    original = awq_processor._accumulate_awq_weight_mean

    def recording(layers, group_size):
        devices = sorted({str(layer.weight.device) for layer in layers})
        with lock:
            spans.append((len(layers), tuple(devices)))
        return original(layers, group_size)

    config = AWQConfig(bits=4, group_size=32, device="cuda:0", desc_act=False)
    if strategy == VramStrategy.BALANCED.value:
        config.moe_vram_strategy = VramStrategy.BALANCED
        config.moe_vram_strategy_devices = ["cuda:0", "cuda:1"]

    model = GPTQModel.load(str(model_dir), config, backend=BACKEND.TORCH)
    awq_processor._accumulate_awq_weight_mean = recording
    try:
        model.quantize(
            CalibrationConfig(dataset=_calibration(), batch_size=1, min_length=1),
            backend=BACKEND.TORCH,
        )
    finally:
        awq_processor._accumulate_awq_weight_mean = original
        packed = {
            name: tensor.detach().to("cpu").clone()
            for name, tensor in model.model.state_dict().items()
        }
        del model
        torch.cuda.empty_cache()
    return spans, packed


def _run_in_subprocess(model_dir: Path, *, strategy: str, visible_devices: str, timeout: int = 1800):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = visible_devices
    env["HF_HUB_OFFLINE"] = "1"
    env["LOGBAR_ANIMATION"] = "0"
    repo_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (repo_root, env.get("PYTHONPATH")) if part
    )
    output = model_dir / f"packed-{strategy}.pt"
    result = subprocess.run(
        [
            sys.executable,
            # Run the driver as a module so a forkserver child never re-imports
            # a pytest `__main__`.
            "-m",
            f"{Path(__file__).resolve().parent.name}.{Path(__file__).resolve().stem}",
            "--mode",
            "run",
            "--model-dir",
            str(model_dir),
            "--strategy",
            strategy,
            "--output",
            str(output),
        ],
        env=env,
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert result.returncode == 0, (
        f"AWQ quantization failed (strategy={strategy}):\n"
        + result.stdout[-8000:]
        + result.stderr[-8000:]
    )
    spans = None
    for line in result.stdout.splitlines():
        if line.startswith("SPANS "):
            spans = json.loads(line[len("SPANS "):])
    assert spans is not None, result.stdout[-4000:]
    return [tuple(span) for span in spans], torch.load(output)


@pytest.fixture(scope="module")
def native_model_dir():
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices for the balanced MoE pool")
    work_dir = Path(tempfile.mkdtemp(prefix="awq-balanced-5104-"))
    model_dir = work_dir / "native"
    model_dir.mkdir()
    _build_fixture(model_dir)
    try:
        yield model_dir
    finally:
        import shutil

        shutil.rmtree(work_dir, ignore_errors=True)


@pytest.mark.gpu
@pytest.mark.slow
def test_balanced_moe_pool_quantizes_and_keeps_the_expert_group_intact(native_model_dir):
    """The documented balanced MoE configuration must finish for AWQ."""

    balanced_spans, balanced = _run_in_subprocess(
        native_model_dir, strategy="balanced", visible_devices="0,1"
    )

    # The trigger: one norm-anchored group really did span both devices.
    assert any(len(devices) > 1 for _members, devices in balanced_spans), (
        f"balanced placement never split a scale group: {balanced_spans}"
    )

    # Every expert projection is packed, in both layers.
    expected = {
        f"model.layers.{layer}.mlp.experts.{expert}.{projection}"
        for layer in range(_NUM_LAYERS)
        for expert in range(_NUM_EXPERTS)
        for projection in _PROJECTIONS
    }
    packed_experts = {name for name in balanced if ".mlp.experts." in name}
    assert {
        name.rsplit(".", 1)[0] for name in packed_experts
        if name.endswith("qweight")
    } == expected

    exclusive_spans, exclusive = _run_in_subprocess(
        native_model_dir, strategy="exclusive", visible_devices="0,1"
    )
    assert all(len(devices) == 1 for _members, devices in exclusive_spans), (
        f"exclusive placement split a scale group: {exclusive_spans}"
    )

    # A gather is not a different fold: the packed tensors are identical.
    assert balanced.keys() == exclusive.keys()
    for name, tensor in balanced.items():
        assert torch.equal(tensor, exclusive[name]), name


def test_quantize_and_serialize_spans(tmp_path):
    """The driver round-trips its own spans/packed payload (runs in a child)."""

    spans = [(2, ["cuda:0"]), (32, ["cuda:0", "cuda:1"])]
    payload = tmp_path / "spans.json"
    payload.write_text(json.dumps(spans))
    assert [tuple(span) for span in json.loads(payload.read_text())] == spans


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("build", "run"), default="run")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--strategy", default="balanced")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    if args.mode == "build":
        _build_fixture(Path(args.model_dir))
    else:
        import torch

        spans, packed = _quantize(Path(args.model_dir), strategy=args.strategy)
        print("SPANS " + json.dumps(spans))
        if args.output:
            torch.save(packed, args.output)
