# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# -- do not touch
import os


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

import os  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import unittest  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402
import torch.utils.benchmark as benchmark  # noqa: E402
from logbar import LogBar  # noqa: E402
from parameterized import parameterized  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from gptqmodel import BACKEND, GPTQModel  # noqa: E402  # ensures monkey patches run before Triton import


log = LogBar.shared()

MODEL_ID = "/monster/data/model/Llama-7B-GPTQ"
DATASET_ID = "timdettmers/openassistant-guanaco"
LEARNING_RATE = 3e-5
MAX_SEQ_LEN = 10
BATCH_SIZE = 5
NUM_TRAIN_STEPS = 10

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def benchmark_forward(
        fn,
        *inputs,
        dtype=torch.dtype,
        repeats="auto",
        desc="",
        verbose=True,
        **kwinputs,
):
    if verbose:
        log.info(desc, f"- Forward pass ({dtype})")

    t = benchmark.Timer(
        stmt="fn(*inputs, **kwinputs)",
        globals={"fn": fn, "inputs": inputs, "kwinputs": kwinputs},
        num_threads=torch.get_num_threads(),
    )
    if repeats == "auto":
        m = t.blocked_autorange()
    else:
        m = t.timeit(repeats)
    if verbose:
        log.info(m)
    return t, m


def get_model_and_tokenizer(
        model_id=MODEL_ID,
        dtype: torch.dtype = None,
        **model_kwargs,
):
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        use_fast=True,
    )
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = GPTQModel.load(
        model_id,
        dtype=dtype,
        **model_kwargs,
    )

    return model, tokenizer


class TestTriton(unittest.TestCase):
    @parameterized.expand(
        [
            (torch.float16),
            (torch.bfloat16)
        ]
    )
    def test_triton_qlinear(self, dtype=torch.dtype):
        ref_model, _ = get_model_and_tokenizer(
            model_id=MODEL_ID,
            backend=BACKEND.TRITON,
            dtype=dtype,
        )

        hidden_size = ref_model.model.model.embed_tokens.weight.shape[1]
        test_data = torch.randn((1, 2048, hidden_size), dtype=dtype).cuda()

        qlinear_ref = ref_model.model.model.layers[0].self_attn.q_proj
        log.info(f"model dtype: {ref_model.model.dtype}")

        ref_out = qlinear_ref(test_data)  # noqa: F841

        _, measure_triton = benchmark_forward(qlinear_ref, test_data, dtype=dtype, desc="Triton", verbose=True)


######## test_triton_autotune_threads.py ######

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - optional dependency
    triton = None
    tl = None


@pytest.mark.skipif(triton is None, reason="Triton is not installed")
def test_triton_autotune_threads_cuda():
    gil_enabled = getattr(sys, "_is_gil_enabled", lambda: True)()
    if gil_enabled:
        pytest.skip("Requires running with PYTHON_GIL=0")
    if not torch.cuda.is_available():
        pytest.skip("CUDA backend required for Triton autotune threading test")

    device = "cuda"
    N = 8192
    configs = [
        triton.Config(kwargs={"BLOCK": 128}, num_warps=2),
        triton.Config(kwargs={"BLOCK": 256}, num_warps=4),
    ]

    @triton.autotune(configs=configs, key=["N"])
    @triton.jit
    def copy_kernel(dst, src, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < N
        values = tl.load(src + offsets, mask=mask)
        tl.store(dst + offsets, values, mask=mask)

    def grid(meta):
        return (triton.cdiv(N, meta["BLOCK"]),)

    num_threads = 8
    sync_ready = threading.Barrier(num_threads + 1)
    sync_start = threading.Barrier(num_threads + 1)
    errors = []

    def worker():
        dst = torch.empty(N, device=device, dtype=torch.float32)
        src = torch.randn_like(dst)
        sync_ready.wait()
        sync_start.wait()
        try:
            for _ in range(4):
                dst.zero_()
                copy_kernel[grid](dst, src, N)
        except Exception as exc:  # pragma: no cover - captured for assertion
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    for thread in threads:
        thread.start()

    sync_ready.wait()
    sync_start.wait()

    for thread in threads:
        thread.join()

    assert not errors
