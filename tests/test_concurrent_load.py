# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Concurrent ``GPTQModel.load()`` must not corrupt the loaded models.

Issue #4129: transformers' ``from_pretrained`` saves/sets/restores the
process-wide default dtype and builds parameters through meta-device and
no-init contexts. Two dense loads in different threads interleaved those
changes, so a model could come back with mixed fp16/fp32 parameters, a meta
``lm_head`` and untied embeddings. The loaders now serialize model construction;
this test loads two models concurrently and compares the result with a serial
load, round by round, under both interpreter modes.

A single passing round proves nothing for a race, so the child runs several
rounds and fails on the first corrupted load.
"""

import os
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "Qwen2.5": "/monster/data/model/Qwen2.5-0.5B-Instruct",
    "Qwen3": "/monster/data/model/Qwen3-0.6B",
}
ROUNDS = 6

_CHILD = r"""
import sys
import threading
from collections import Counter

import torch

from gptqmodel import GPTQConfig, GPTQModel

MODELS = {models!r}
ROUNDS = {rounds}


def load(path):
    return GPTQModel.load(
        path,
        GPTQConfig(bits=4, group_size=128, offload_to_disk=False),
        dtype=torch.float16,
        device="cpu",
    )


def inspect(model):
    params = {{name: p for name, p in model.model.named_parameters() if p.is_floating_point()}}
    dtypes = Counter(str(p.dtype) for p in params.values())
    meta = [name for name, p in params.items() if p.is_meta]
    tied = (
        model.model.lm_head.weight.data_ptr()
        == model.model.model.embed_tokens.weight.data_ptr()
    )
    return dtypes, meta, tied, torch.get_default_dtype()


def bad_reasons(state):
    dtypes, meta, tied, default_dtype = state
    reasons = []
    if set(dtypes) != {{"torch.float16"}}:
        reasons.append(f"dtypes={{dict(dtypes)}}")
    if meta:
        reasons.append(f"meta={{meta[:3]}}")
    if not tied:
        reasons.append("UNTIED")
    if default_dtype != torch.float32:
        reasons.append(f"default_dtype_after={{default_dtype}}")
    return reasons


def main():
    print(f"gil_enabled={{sys._is_gil_enabled()}}", flush=True)
    bad_rounds = 0
    for rnd in range(ROUNDS):
        results = {{}}
        errors = []

        def worker(name, path):
            try:
                results[name] = inspect(load(path))
            except BaseException as exc:  # noqa: BLE001
                errors.append((name, repr(exc)[:200]))

        threads = [threading.Thread(target=worker, args=item) for item in MODELS.items()]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=600)

        bad = {{name: bad_reasons(state) for name, state in results.items()}}
        bad = {{name: reasons for name, reasons in bad.items() if reasons}}
        if bad or errors:
            bad_rounds += 1
        print(f"ROUND {{rnd}} {{'BAD' if bad or errors else 'ok'}} {{bad}} errors={{errors}}", flush=True)
        torch.set_default_dtype(torch.float32)
    print(f"bad_rounds={{bad_rounds}}/{{ROUNDS}}", flush=True)
    return 0 if bad_rounds == 0 else 1


sys.exit(main())
"""


def _free_threaded() -> bool:
    return sysconfig.get_config_var("Py_GIL_DISABLED") == 1


def _models_available() -> bool:
    return all(Path(path).exists() for path in MODELS.values())


@pytest.mark.skipif(not _free_threaded(), reason="requires a free-threaded Python build")
@pytest.mark.skipif(not _models_available(), reason="concurrent-load checkpoints are not available")
def test_concurrent_dense_loads_are_serialized():
    code = _CHILD.format(models=MODELS, rounds=ROUNDS)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env["HIP_VISIBLE_DEVICES"] = "0"

    for gil in ("0", "1"):
        env["PYTHON_GIL"] = gil
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        assert result.returncode == 0, (
            f"PYTHON_GIL={gil}: concurrent loads corrupted a model\n"
            f"stdout:\n{result.stdout[-6000:]}\nstderr:\n{result.stderr[-4000:]}"
        )
        assert f"bad_rounds=0/{ROUNDS}" in result.stdout
        assert f"gil_enabled={'False' if gil == '0' else 'True'}" in result.stdout
