# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Data-parallel replicas must not share subset early-stop state (#4904).

``ForwardExecutor.run_parallel`` forwards one replica of the layer per device
on a device-dedicated worker thread. The replicas are deep copies, but a deep
copy keeps the closure objects registered by the subset stage, and the subset
stage used to keep a single ``pending_members`` set plus a stop flag on the
subset's *original* modules. One replica could therefore drain another
replica's pending set and flip ``forward_hook_last`` on a module the other
replica was still forwarding; ``HookedLinear`` then raised ``StopForward`` and
every routed expert after that member silently lost the batch.

This test runs the tiny Qwen3-MoE fixture from
``tests/test_checkpoint_quantization.py`` on two GPUs with the default
``auto_forward_data_parallel=True``. Routing is overridden so every expert sees
every token, which makes ``nsamples`` a direct count of the calibration tokens
that reached each expert projection: 4 rows x 32 tokens = 128 for all of them.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


_NUM_LAYERS = 2
_NUM_EXPERTS = 4
_ROWS = 4
_TOKENS_PER_ROW = 32
_EXPECTED_NSAMPLES = _ROWS * _TOKENS_PER_ROW
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
}


def _build_fixture(model_dir: Path) -> None:
    """Persist the tiny two-layer Qwen3-MoE fixture used by the issue."""

    import kernels  # noqa: F401  (must win over tests/kernels for transformers)
    import torch

    # Imported lazily: the subprocess driver below must not import the test
    # helpers (its ``tests/`` directory shadows the installed ``kernels``
    # package that transformers imports).
    tests_dir = str(Path(__file__).resolve().parent)
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)
    from test_checkpoint_quantization import build_native_model
    from test_tiny_moe_quant_smoke import _build_local_tokenizer

    torch.manual_seed(1234)
    model = build_native_model(
        "qwen3_moe", dict(_CONFIG_ARGS), moe_intermediate_size=32
    )
    model.save_pretrained(model_dir)
    _build_local_tokenizer(model_dir)


def _calibration():
    """Four deterministic rows of 32 tokens, as in the issue's repro."""

    import torch

    torch.manual_seed(6789)
    return [
        {
            "input_ids": torch.randint(3, 100, (1, _TOKENS_PER_ROW)),
            "attention_mask": torch.ones(1, _TOKENS_PER_ROW, dtype=torch.long),
        }
        for _ in range(_ROWS)
    ]


def _collect_nsamples(
    model_dir: Path, *, device: str, auto_forward_data_parallel: bool
):
    """Quantize the fixture and return the finalized ``GPTQ.nsamples`` per module."""

    import threading

    import torch

    from gptqmodel import BACKEND, CalibrationConfig, GPTQModel, QuantizeConfig
    from gptqmodel.quantization.config import ExpertsRoutingOverride, MoEConfig
    from gptqmodel.quantization.gptq import GPTQ

    torch.manual_seed(6789)
    config = QuantizeConfig(
        bits=4,
        group_size=32,
        desc_act=False,
        device=device,
        auto_forward_data_parallel=auto_forward_data_parallel,
        offload_to_disk=False,
        fallback=None,
        # Route every token to every expert: a projection that sees fewer than
        # the full calibration set lost a batch to the early stop, it was not
        # skipped by the router.
        moe=MoEConfig(routing=ExpertsRoutingOverride()),
    )
    model = GPTQModel.load(
        str(model_dir), quantize_config=config, backend=BACKEND.TORCH
    )

    recorded = {}
    lock = threading.Lock()
    original_finalize = GPTQ.finalize_hessian

    def record_finalize(task, *args, **kwargs):
        hessian = original_finalize(task, *args, **kwargs)
        name = task._named_module.full_name
        with lock:
            recorded[name] = task.nsamples
        return hessian

    GPTQ.finalize_hessian = record_finalize
    try:
        model.quantize(
            CalibrationConfig(
                dataset=_calibration(),
                batch_size=1,
                min_length=1,
            ),
            backend=BACKEND.TORCH,
        )
    finally:
        GPTQ.finalize_hessian = original_finalize
        del model
        torch.cuda.empty_cache()
    return recorded


def _expert_nsamples(recorded):
    return {
        name: nsamples
        for name, nsamples in recorded.items()
        if ".mlp.experts." in name and name.endswith(_PROJECTIONS)
    }


def _assert_every_projection_complete(recorded, *, label):
    experts = _expert_nsamples(recorded)
    expected_names = {
        f"model.layers.{layer}.mlp.experts.{expert}.{projection}"
        for layer in range(_NUM_LAYERS)
        for expert in range(_NUM_EXPERTS)
        for projection in _PROJECTIONS
    }
    observed_names = set(experts)
    assert observed_names == expected_names, (
        f"{label}: missing expert projections: "
        f"{sorted(expected_names - observed_names)}"
    )
    incomplete = {
        name: value
        for name, value in sorted(experts.items())
        if value != _EXPECTED_NSAMPLES
    }
    assert not incomplete, (
        f"{label}: routed experts lost calibration tokens: "
        f"{incomplete}"
    )
    for layer in range(_NUM_LAYERS):
        for expert in range(_NUM_EXPERTS):
            prefix = f"model.layers.{layer}.mlp.experts.{expert}"
            gate = experts[f"{prefix}.gate_proj"]
            up = experts[f"{prefix}.up_proj"]
            down = experts[f"{prefix}.down_proj"]
            assert gate == up == down == _EXPECTED_NSAMPLES, (
                f"{label}: {prefix} gate/up/down saw {gate}/{up}/{down} tokens"
            )


def _run_in_subprocess(
    model_dir: Path,
    *,
    mode: str = "run",
    visible_devices: str,
    auto_forward_data_parallel: bool = True,
    timeout: int = 900,
):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = visible_devices
    env["HF_HUB_OFFLINE"] = "1"
    env["LOGBAR_ANIMATION"] = "0"
    env["GPTQ_BYPASS_SAFETY_CHECKS"] = "1"
    repo_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (repo_root, env.get("PYTHONPATH")) if part
    )
    result = subprocess.run(
        [
            sys.executable,
            # Run the driver as a module, not as a script: a forkserver child
            # re-imports the parent's ``__main__``, and under a pytest launcher
            # that would start a second, nested pytest run of this file.
            "-m",
            f"{Path(__file__).resolve().parent.name}.{Path(__file__).resolve().stem}",
            "--mode",
            mode,
            "--model-dir",
            str(model_dir),
            "--device",
            "cuda:0",
            "--auto-dp",
            "1" if auto_forward_data_parallel else "0",
        ],
        env=env,
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert result.returncode == 0, (
        f"{mode} subprocess failed (CUDA_VISIBLE_DEVICES={visible_devices}, "
        f"auto_forward_data_parallel={auto_forward_data_parallel}):\n"
        + result.stdout[-8000:]
        + result.stderr[-8000:]
    )
    for line in result.stdout.splitlines():
        if line.startswith("NSAMPLES "):
            return json.loads(line[len("NSAMPLES ") :])
    if mode == "build":
        persisted = (model_dir / "config.json").is_file()
        assert persisted, (
            "build subprocess did not persist the fixture at "
            f"{model_dir}:\n{result.stdout[-4000:]}{result.stderr[-4000:]}"
        )
        return None
    raise AssertionError(
        "quantization subprocess did not report NSAMPLES:\n"
        + result.stdout[-8000:]
        + result.stderr[-8000:]
    )


@pytest.fixture(scope="module")
def native_model_dir():
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("requires two CUDA devices for the data-parallel forward")
    # Keep the fixture outside pytest's basetemp: a forkserver child of this
    # interpreter re-imports ``__main__`` (the pytest launcher in the harness),
    # and a nested pytest run would clean the shared basetemp mid-test.
    work_dir = Path(tempfile.mkdtemp(prefix="dp-subset-4904-"))
    model_dir = work_dir / "native"
    # Build in the driver subprocess: the parent pytest process must not start
    # the model stack (a forkserver child re-imports its ``__main__``).
    _run_in_subprocess(model_dir, mode="build", visible_devices="0")
    try:
        yield model_dir
    finally:
        import shutil

        shutil.rmtree(work_dir, ignore_errors=True)


@pytest.mark.gpu
@pytest.mark.slow
def test_dp_replicas_keep_every_routed_token(native_model_dir):
    """Every expert projection must see the full calibration set on two GPUs."""

    first = _run_in_subprocess(
        native_model_dir, visible_devices="0,1", auto_forward_data_parallel=True
    )
    _assert_every_projection_complete(first, label="2 GPUs, default DP")

    # The amount of loss varied between runs in the issue. Two back-to-back
    # default runs must also agree on every module's sample count.
    second = _run_in_subprocess(
        native_model_dir, visible_devices="0,1", auto_forward_data_parallel=True
    )
    assert first == second, (
        "2-GPU data-parallel nsamples are not deterministic between runs: "
        f"{ {name: (first.get(name), second.get(name)) for name in sorted(set(first) | set(second)) if first.get(name) != second.get(name)} }"
    )


@pytest.mark.gpu
@pytest.mark.slow
def test_single_gpu_and_dp_off_controls_keep_every_routed_token(native_model_dir):
    """The serial paths were already correct and must stay that way."""

    dp_off = _run_in_subprocess(
        native_model_dir, visible_devices="0,1", auto_forward_data_parallel=False
    )
    _assert_every_projection_complete(dp_off, label="2 GPUs, DP off")

    single = _run_in_subprocess(
        native_model_dir, visible_devices="0", auto_forward_data_parallel=True
    )
    _assert_every_projection_complete(single, label="1 GPU")


class _FlaggedModule:
    """Stand-in for a ``HookedLinear`` with the subset stop flag."""

    def __init__(self):
        self.forward_hook_last = False


def _drain(state, *member_names):
    """Discard members in order and return whether each one completed the set."""

    return [state.discard(name) for name in member_names]


def test_early_stop_state_is_per_forward_thread():
    """One replica's worker thread must not drain another replica's members."""

    import threading

    from gptqmodel.looper.stage_subset import _SubsetEarlyStopState

    state = _SubsetEarlyStopState(["a", "b"])
    state.reset()
    completed = _drain(state, "a", "b")
    assert completed == [False, True]

    # A worker thread starts from the full member set even though the calling
    # thread already completed the subset.
    observed = {}

    def worker():
        observed["a"] = state.discard("a")
        observed["b"] = state.discard("b")

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    assert observed == {"a": False, "b": True}

    # The layer pre-hook rebuilds the state for the next batch on this thread.
    state.reset()
    completed = _drain(state, "a", "b")
    assert completed == [False, True]


def test_completing_member_flags_the_module_that_ran():
    """The stop flag must land on the replica's module, not the closure's."""

    from gptqmodel.looper.stage_subset import _mark_subset_member, _SubsetEarlyStopState

    state = _SubsetEarlyStopState(["a", "b"])
    state.reset()
    closure_module = _FlaggedModule()
    first_replica = _FlaggedModule()
    second_replica = _FlaggedModule()
    captured = []

    hook = _mark_subset_member(
        lambda module, inputs, output: captured.append(module), "a", state
    )
    hook(first_replica, None, None)
    assert first_replica.forward_hook_last is False

    hook = _mark_subset_member(
        lambda module, inputs, output: captured.append(module), "b", state
    )
    hook(second_replica, None, None)
    assert second_replica.forward_hook_last is True
    # The other replica and the closure's module must stay unflagged.
    assert first_replica.forward_hook_last is False
    assert closure_module.forward_hook_last is False
    assert captured == [first_replica, second_replica]


def test_replica_member_resolves_the_executing_copy():
    """Names are layer-relative, so the same path names the replica's copy."""

    import torch

    from gptqmodel.looper.stage_subset import _subset_member_for_replica

    class Layer(torch.nn.Module):
        pass

    def build():
        layer = Layer()
        layer.mlp = torch.nn.Module()
        layer.mlp.experts = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
        return layer

    original = build()
    replica = build()
    fallback = object()
    resolved = _subset_member_for_replica(
        replica, original.mlp.experts[0], "mlp.experts.0"
    )
    assert resolved is replica.mlp.experts[0]
    # Absolute subset keys (embeddings, lm_head) are not layer-relative.
    unresolved = _subset_member_for_replica(
        replica, fallback, "model.embed_tokens"
    )
    assert unresolved is fallback


def _main(argv=None):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("build", "run"), default="run")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--auto-dp", type=int, default=1)
    args = parser.parse_args(argv)
    if args.mode == "build":
        _build_fixture(Path(args.model_dir))
        print("BUILT", flush=True)
        return
    recorded = _collect_nsamples(
        Path(args.model_dir),
        device=args.device,
        auto_forward_data_parallel=bool(args.auto_dp),
    )
    print("NSAMPLES " + json.dumps(recorded), flush=True)


if __name__ in ("__main__", "__mp_main__"):
    # When this file runs as the quantization subprocess driver, its own
    # directory is first on sys.path and ``tests/kernels`` would shadow the
    # installed ``kernels`` package that transformers imports. Drop it before
    # any heavy import; the driver needs neither the test helpers nor pytest.
    _here = str(Path(__file__).resolve().parent)
    sys.path[:] = [entry for entry in sys.path if entry not in ("", _here)]

if __name__ == "__main__":
    _main()
