# Copyright 2024-2025 ModelCloud.ai
# Copyright 2024-2025 qubitium@modelcloud.ai
# Contact: qubitium@modelcloud.ai, x.com/qubitium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# -- do not touch
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

import os  # noqa: E402
import unittest  # noqa: E402

import torch  # noqa: E402
import torch.utils.benchmark as benchmark  # noqa: E402
from gptqmodel import BACKEND, GPTQModel  # noqa: E402
from logbar import LogBar  # noqa: E402
from parameterized import parameterized  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

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
        torch_dtype=dtype,
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

        ref_out = qlinear_ref(test_data) # noqa: F841

        _, measure_triton = benchmark_forward(qlinear_ref, test_data, dtype=dtype, desc="Triton", verbose=True)
