# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Cold concurrent AWQ Triton JIT must not abort a free-threaded process.

Issue #4094: with ``PYTHON_GIL=0`` and an empty ``TRITON_CACHE_DIR``, 24 threads
launched the first ``awq_gemm_kernel`` together. libtriton is not free-thread
safe, so its nanobind refcounts collided and the process died with
``attempted to delete an unknown instance`` (exit code 134). The fix serializes
compilation of GPT-QModel's own kernels (``gptqmodel.utils.triton_jit``); this test reproduces the cold launch
in a child process and asserts a clean exit. It is repeated because a single
pass is not evidence for a race.
"""

import os
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest


MODEL_ID = "/monster/data/model/AWQ-Llama-3.2-1B-g128-gemm"
REPO_ROOT = Path(__file__).resolve().parents[1]

_CHILD = r"""
import os, sys, tempfile, threading, time
cache = tempfile.mkdtemp(prefix="triton_cold_")
os.environ["TRITON_CACHE_DIR"] = cache
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import torch
assert sys._is_gil_enabled() is False, "child did not run free-threaded"
from transformers import AutoTokenizer
from gptqmodel import BACKEND, GPTQModel

MID = {model_id!r}
N = {threads}
model = GPTQModel.load(MID, backend=BACKEND.AWQ_GEMM_TRITON)
tokenizer = AutoTokenizer.from_pretrained(MID)
inputs = tokenizer(["Hello, my name is"], return_tensors="pt")
barrier = threading.Barrier(N)
errors = []

def work(index):
    try:
        barrier.wait(timeout=120)
        model.generate(**inputs, max_new_tokens=8, do_sample=False)
    except BaseException as exc:
        errors.append((index, type(exc).__name__, str(exc)[:200]))

threads = [threading.Thread(target=work, args=(index,), daemon=True) for index in range(N)]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join(timeout=300)
alive = [thread.name for thread in threads if thread.is_alive()]
if errors or alive:
    print(f"errors={{errors}} alive={{alive}}", file=sys.stderr)
    os._exit(1)
print(f"cold-jit ok threads={{N}} cache={{cache}}")
os._exit(0)
"""


def _free_threaded() -> bool:
    return sysconfig.get_config_var("Py_GIL_DISABLED") == 1


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


@pytest.mark.skipif(not _free_threaded(), reason="requires a free-threaded Python build")
@pytest.mark.skipif(not Path(MODEL_ID).exists(), reason=f"{MODEL_ID} not available")
@pytest.mark.skipif(not _cuda_available(), reason="requires an accelerator")
def test_cold_concurrent_awq_triton_jit_exits_cleanly():
    env = os.environ.copy()
    env["PYTHON_GIL"] = "0"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    code = _CHILD.format(model_id=MODEL_ID, threads=24)
    for attempt in range(2):
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
        )
        assert result.returncode == 0, (
            f"attempt {attempt}: cold concurrent AWQ Triton JIT exited "
            f"{result.returncode}\nstdout:\n{result.stdout[-4000:]}\nstderr:\n{result.stderr[-4000:]}"
        )
