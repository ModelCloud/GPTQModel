# -- do not touch
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

import os  # noqa: E402
import unittest  # noqa: E402

import torch  # noqa: E402
import torch.utils.benchmark as benchmark  # noqa: E402
from gptqmodel import BACKEND, GPTQModel  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

MODEL_ID = "TheBloke/Llama-7B-GPTQ"
DATASET_ID = "timdettmers/openassistant-guanaco"
LEARNING_RATE = 3e-5
MAX_SEQ_LEN = 10
BATCH_SIZE = 5
NUM_TRAIN_STEPS = 10

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def benchmark_forward(
    fn,
    *inputs,
    repeats="auto",
    desc="",
    verbose=True,
    amp=False,
    amp_dtype=torch.float16,
    **kwinputs,
):
    if verbose:
        print(desc, "- Forward pass")

    def amp_wrapper(*inputs, **kwinputs):
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp):
            fn(*inputs, **kwinputs)

    t = benchmark.Timer(
        stmt="fn_amp(*inputs, **kwinputs)",
        globals={"fn_amp": amp_wrapper, "inputs": inputs, "kwinputs": kwinputs},
        num_threads=torch.get_num_threads(),
    )
    if repeats == "auto":
        m = t.blocked_autorange()
    else:
        m = t.timeit(repeats)
    if verbose:
        print(m)
    return t, m


def get_model_and_tokenizer(
    model_id=MODEL_ID,
    **model_kwargs,
):
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        use_fast=True,
    )
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = GPTQModel.from_quantized(
        model_id,
        **model_kwargs,
    )

    return model, tokenizer


class TestTriton(unittest.TestCase):
    def test_triton_qlinear(self):
        ref_model, _ = get_model_and_tokenizer(
            model_id=MODEL_ID,
            backend=BACKEND.TRITON,
        )

        hidden_size = ref_model.model.model.embed_tokens.weight.shape[1]
        test_data = torch.randn((1, 2048, hidden_size), dtype=torch.float16).cuda()

        qlinear_ref = ref_model.model.model.layers[0].self_attn.q_proj

        ref_out = qlinear_ref(test_data) # noqa: F841

        _, measure_triton = benchmark_forward(qlinear_ref, test_data, desc="Triton", verbose=True)
