# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Opt-in full Llama 3.2 1B staged GSQ and GSM8K Platinum check.

This uses all 16 decoder blocks and all seven linear projections per block.
Smoke settings use 32 nm-calibration examples, one training epoch, one Q/K
step, 128 GSM8K Platinum rows and 96 generated tokens. Set
GPTQMODEL_GSQ_STAGED_EPOCHS and GPTQMODEL_GSQ_STAGED_QK_STEPS to exercise a
longer schedule. With the exact six concatenated training sequences in this
smoke recipe, the 16-block GPTQ initializer-only run scored 36/128 and the
one-epoch staged GPTQ run scored 31/128; two epochs scored 28/128. This test
detects large quality failures; it does not claim that the short GSQ schedule
improves accuracy.
The earlier 59/128 GPTQ-only score used two decoder blocks and different
activation-group settings, so it is not a matched baseline here.
"""

import os
from pathlib import Path

import pytest
import torch

from model_test import ModelTest

from gptqmodel import GPTQModel
from gptqmodel.models._const import DEVICE
from gptqmodel.nn_modules.qlinear.torch import TorchLinear
from gptqmodel.nn_modules.qlinear.torch_awq import AwqTorchLinear
from gptqmodel.quantization import AWQConfig, FORMAT, GPTQConfig
from gptqmodel.utils.backend import BACKEND
from tests.eval import evaluate, get_eval_task_results


@pytest.mark.parametrize("method", ["gptq", "awq"])
def test_llama3_2_1b_staged_gsq_gsm8k_platinum(method):
    if os.environ.get("GPTQMODEL_RUN_LLAMA3_2_GSQ_STAGED_E2E") != "1":
        pytest.skip("Set GPTQMODEL_RUN_LLAMA3_2_GSQ_STAGED_E2E=1 for full model training and evaluation")
    if not torch.cuda.is_available():
        pytest.skip("This full-model quality check requires CUDA")

    source = Path("/monster/data/model/Llama-3.2-1B-Instruct")
    assert source.is_dir(), f"Missing Llama 3.2 1B source at {source}"
    options = dict(enabled=True,
                   epochs=int(os.environ.get("GPTQMODEL_GSQ_STAGED_EPOCHS", "1")),
                   qk_steps=int(os.environ.get("GPTQMODEL_GSQ_STAGED_QK_STEPS", "1")),
                   batch_size=1, microbatch_size=1)
    if method == "gptq":
        config = GPTQConfig(bits=4, group_size=128, sym=True, desc_act=False,
                            act_group_aware=False, format=FORMAT.GPTQ_V2,
                            device=DEVICE.CUDA, offload_to_disk=False,
                            gsq_training=options)
        packed_type = TorchLinear
    else:
        config = AWQConfig(bits=4, group_size=128, sym=False, format=FORMAT.GEMM,
                           device=DEVICE.CUDA, offload_to_disk=False,
                           gsq_training=options)
        packed_type = AwqTorchLinear

    model = GPTQModel.load(str(source), quantize_config=config, dtype=torch.bfloat16,
                           attn_implementation="eager")
    model.model.to("cuda:0")
    calibration = ModelTest.load_dataset(rows=32)
    result = model.quantize(calibration, calibration_concat_size=2048,
                            calibration_sort="desc", backend=BACKEND.TORCH)
    assert model.quantized
    assert len(model.model.model.layers) == 16
    for layer in model.model.model.layers:
        for name in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                     "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
            assert isinstance(layer.get_submodule(name), packed_type)
    if method == "gptq":
        assert len(result["gsq_training"]) == 16
    else:
        assert len(config.meta["gsq_training_runs"]) == 16

    save_root = Path(os.environ.get("GPTQMODEL_GSQ_STAGED_SAVE_ROOT", "/tmp"))
    output = save_root / f"llama3_2_1b_{method}_gsq_staged"
    model.save(str(output))
    del model
    torch.cuda.empty_cache()

    raw = evaluate(
        model_or_id_or_path=str(output), tasks=["gsm8k_platinum_cot"],
        backend=BACKEND.TORCH, batch_size="auto", apply_chat_template=True,
        model_args={"dtype": "bfloat16", "attn_implementation": "sdpa", "device": "cuda:0", "seed": 898},
        gen_kwargs="do_sample=false,temperature=0.0,top_p=1.0,top_k=50",
        suite_kwargs={"batch_size": 8, "max_rows": 128, "max_new_tokens": 96, "stream": True},
    )
    score = get_eval_task_results(raw)["gsm8k_platinum_cot"]["acc,num"]
    print(f"{method} staged GSQ GSM8K Platinum: {score:.8f} ({score * 128:.0f}/128)")
    minimum = float(os.environ.get("GPTQMODEL_GSQ_STAGED_MIN_GSM8K", "0.20"))
    assert score >= minimum, f"{method} staged GSQ GSM8K Platinum {score:.4f} < {minimum:.4f}"
