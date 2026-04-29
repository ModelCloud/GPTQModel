# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import tempfile

import pytest
from datasets import load_dataset
from logbar import LogBar

from gptqmodel import GPTAQConfig, GPTQModel, QuantizeConfig
from gptqmodel.quantization import FORMAT
from tests.eval import evaluate, format_eval_result_table, get_eval_task_metrics


pytestmark = [pytest.mark.model, pytest.mark.slow]


log = LogBar.shared()

MODEL_ID = "/monster/data/model/Llama-3.1-8B-Instruct"
CFG_BITS = 4
CFG_GROUPSIZE = 128
CFG_V2 = True
INPUTS_MAX_LENGTH = 2048 # in tokens
QUANT_SAVE_PATH = f"/monster/data/model/gptq_v2_{CFG_V2}_bit_{CFG_BITS}_gpsize_{CFG_GROUPSIZE}_llama_3.1_8B_Instruct"

RAND_SEED = 898

EVAL_ONLY = True
EVAL_APPLY_CHAT_TEMPLATE = True

def get_calib_data(tokenizer, rows: int):
    # calibration_dataset = load_dataset(
    #     "allenai/c4",
    #     data_files="en/c4-train.00000-of-01024.json.gz",
    #     split="train"
    # )

    calibration_dataset = load_dataset(
        "json",
        data_files="/monster/data/model/dataset/c4-train.00000-of-01024.json.gz",
        split="train")

    datas = []
    for index, sample in enumerate(calibration_dataset):
        tokenized = tokenizer(sample["text"])
        if len(tokenized.data['input_ids']) <= INPUTS_MAX_LENGTH:
            datas.append(tokenized)
            if len(datas) >= rows:
                break

    return datas

quant_config = QuantizeConfig(
    bits=CFG_BITS,
    group_size=CFG_GROUPSIZE,
    format=FORMAT.GPTQ,
    desc_act=True,
    sym=True,
    gptaq=GPTAQConfig() if CFG_V2 else None,
)

def _run_simple_quant_eval():
    """Run the legacy simple-quant flow as a real pytest workload."""
    log.info(f"QuantConfig: {quant_config}")

    if not EVAL_ONLY:
        log.info(f"Save Path: {QUANT_SAVE_PATH}")

        # load un-quantized native model
        model = GPTQModel.load(MODEL_ID, quant_config)

        # load calibration data
        calibration_dataset = get_calib_data(tokenizer=model.tokenizer, rows=256)

        model.quantize(calibration_dataset, batch_size=1)

        model.save(QUANT_SAVE_PATH)
        log.info(f"Quant Model Saved to: {QUANT_SAVE_PATH}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        results = evaluate(
            QUANT_SAVE_PATH,
            tasks=["gsm8k_cot"],  #, "gsm8k_platinum_cot"],
            apply_chat_template=True,
            output_path=tmp_dir,
        )

        print(format_eval_result_table(results))

        metrics = get_eval_task_metrics(results, "gsm8k_cot")
        filtered_metrics = {
            metric: value
            for metric, value in metrics.items()
            if metric != "alias" and "stderr" not in metric
        }

        value = filtered_metrics['acc,num']
        expected = 0.7998
        diff_pct = (value / expected) * 100
        floor_pct = 0.05
        ceil_pct = 0.10
        negative_pct = 100 * (1 - floor_pct)
        positive_pct = 100 * (1 + ceil_pct)

        assert negative_pct <= diff_pct <= positive_pct, (f"gsm8k_cot:acc,num: `{value}` vs "
                                                          f"expected `{expected}`, diff {diff_pct:.2f}% is out of the "
                                                          f"expected range [{negative_pct}-{positive_pct}%]")


def test_simple_quant():
    """Keep the simple-quant regression runnable under pytest collection."""
    _run_simple_quant_eval()
