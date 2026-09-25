# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Opt-in W3 Llama 3.2 1B test with the paper's Llama calibration budget.

The existing nm calibration parquet supplies disjoint source rows for 4,096
train, 128 validation, and 512 GPTQ sequences of 4,096 tokens. Each source
split repeats its own rows to reach that budget; the source has fewer unique
tokens than the paper's FineWeb-Edu corpus. The paper's 20 epochs, batch 64,
microbatch 2, 2,000 Q/K steps and W3/group128 schedule are used. The model
and GSM8K Platinum evaluation remain the requested Llama 3.2 1B check.
The complete run is expensive and requires a large disk capture directory.
"""

import os
from pathlib import Path

import pytest
import torch
from gsq_paper_data import (
    GPTQ_SEQUENCES,
    SEQUENCE_LENGTH,
    TRAIN_SEQUENCES,
    VALIDATION_SEQUENCES,
    nm_calibration_documents,
    split_documents,
)

from gptqmodel import GPTQModel
from gptqmodel.looper.gsq_training_model import (
    finalize_llama_gsq_wrapper,
    quantize_llama_gsq_model,
)
from gptqmodel.models._const import DEVICE
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.quantization import FORMAT, GPTQConfig, GSQTrainingConfig
from gptqmodel.utils.backend import BACKEND
from tests.eval import evaluate, get_eval_task_results


def test_nm_paper_budget_split_keeps_source_rows_disjoint():
    class Tokenizer:
        def __call__(self, text, *, return_tensors):
            assert return_tensors is None
            return {"input_ids": [int(value) for value in text]}

    rows = (str(index) * 4 for index in range(6))
    train, validation, gptq = split_documents(
        rows, Tokenizer(), sequence_length=4,
        train_sequences=4, validation_sequences=2, gptq_sequences=2)
    assert (len(train), len(validation), len(gptq)) == (4, 2, 2)
    assert all(len(entry["input_ids"]) == 4 for split in (train, validation, gptq) for entry in split)
    token_sets = [{token for entry in split for token in entry["input_ids"]}
                  for split in (train, validation, gptq)]
    assert all(not left.intersection(right) for index, left in enumerate(token_sets)
               for right in token_sets[index+1:])


@pytest.mark.parametrize("staged", [False, True], ids=["gptq-initializer", "gsq"])
def test_llama3_2_1b_gsq_paper_budget_w3(staged):
    if os.environ.get("GPTQMODEL_RUN_GSQ_PAPER_W3_E2E") != "1":
        pytest.skip("Set GPTQMODEL_RUN_GSQ_PAPER_W3_E2E=1 for the full paper-budget run")
    if not torch.cuda.is_available():
        pytest.skip("Paper-budget GSQ needs a CUDA device")

    source = Path("/monster/data/model/Llama-3.2-1B-Instruct")
    assert source.is_dir(), f"Missing Llama 3.2 1B source at {source}"
    recipe = GSQTrainingConfig(enabled=staged)
    assert (recipe.epochs, recipe.batch_size, recipe.microbatch_size, recipe.qk_steps) == (20, 64, 2, 2000)
    assert (recipe.assignment_lr, recipe.scale_lr, recipe.weight_decay, recipe.betas) == (
        1e-4, 5e-5, 1., (.9, .95))
    assert (recipe.temperature, recipe.multiplier, recipe.min_lr, recipe.decay) == (
        (2., .05), (100., 500.), .1, "cosine")
    config = GPTQConfig(bits=3, group_size=128, sym=True, desc_act=False,
                        act_group_aware=False, format=FORMAT.GPTQ_V2,
                        device=DEVICE.CUDA, offload_to_disk=False,
                        gsq_training=recipe)
    wrapper = GPTQModel.load(str(source), quantize_config=config, dtype=torch.bfloat16,
                             attn_implementation="sdpa")
    wrapper.model.to("cuda:0")
    train, validation, gptq = nm_calibration_documents(wrapper.tokenizer)
    assert (len(train), len(validation), len(gptq)) == (
        TRAIN_SEQUENCES, VALIDATION_SEQUENCES, GPTQ_SEQUENCES)
    assert all(len(document["input_ids"]) == SEQUENCE_LENGTH
               for split in (train, validation, gptq) for document in split)
    capture_root = Path(os.environ.get("GPTQMODEL_GSQ_PAPER_CAPTURE_ROOT",
                                       "/monster/data/model/gsq-paper-capture"))
    run = quantize_llama_gsq_model(
        wrapper.model, train if staged else gptq,
        initialization_documents=gptq if staged else None,
        validation_documents=validation if staged else None,
        bits=3, group_size=128, gsq=recipe,
        offload_capture=True, capture_directory=capture_root,
    )
    assert run["state"] == "complete" and len(run["blocks"]) == 16
    assert (run["training_documents"], run["initialization_documents"], run["validation_documents"]) == (
        (TRAIN_SEQUENCES, GPTQ_SEQUENCES, VALIDATION_SEQUENCES) if staged else
        (GPTQ_SEQUENCES, GPTQ_SEQUENCES, 0))
    for layer in wrapper.model.model.layers:
        for name in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                     "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
            assert isinstance(layer.get_submodule(name), TorchLinear)
    finalize_llama_gsq_wrapper(wrapper, run)
    output_root = Path(os.environ.get("GPTQMODEL_GSQ_PAPER_SAVE_ROOT", "/monster/data/model"))
    output = output_root / f"llama3_2_1b_gptq_paper_w3_{'gsq' if staged else 'initializer'}"
    wrapper.save(str(output))
    del wrapper
    torch.cuda.empty_cache()

    raw = evaluate(
        model_or_id_or_path=str(output), tasks=["gsm8k_platinum_cot"],
        backend=BACKEND.TORCH, batch_size="auto", apply_chat_template=True,
        model_args={"dtype": "bfloat16", "attn_implementation": "sdpa", "device": "cuda:0", "seed": 898},
        gen_kwargs="do_sample=false,temperature=0.0,top_p=1.0,top_k=50",
        suite_kwargs={"batch_size": 8, "max_rows": 128, "max_new_tokens": 96, "stream": True},
    )
    score = get_eval_task_results(raw)["gsm8k_platinum_cot"]["acc,num"]
    print(f"paper-budget W3 {'GSQ' if staged else 'GPTQ'} GSM8K Platinum: {score:.8f} ({score * 128:.0f}/128)")
    minimum = float(os.environ.get("GPTQMODEL_GSQ_PAPER_MIN_GSM8K", "0.20"))
    if staged:
        assert score >= minimum
