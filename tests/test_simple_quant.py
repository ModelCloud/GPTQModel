# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import tempfile

from datasets import load_dataset
from logbar import LogBar

from gptqmodel import GPTQModel, QuantizeConfig
from gptqmodel.quantization import FORMAT
from gptqmodel.utils.eval import EVAL


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
    v2=CFG_V2,
)

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

# eval
from lm_eval.utils import make_table


with tempfile.TemporaryDirectory() as tmp_dir:
    results = GPTQModel.eval(
        QUANT_SAVE_PATH,
        tasks=[EVAL.LM_EVAL.GSM8K_COT], #, EVAL.LM_EVAL.GSM8K_PLATINUM_COT],
        apply_chat_template=True,
        random_seed=898,
        output_path= tmp_dir,
    )

    print(make_table(results))
    if "groups" in results:
        print(make_table(results, "groups"))
