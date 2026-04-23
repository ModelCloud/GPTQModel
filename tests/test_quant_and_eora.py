# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# -- do not touch
import os


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

from parameterized import parameterized  # noqa: E402

from gptqmodel.quantization import FORMAT, METHOD  # noqa: E402


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

import tempfile  # noqa: E402
from typing import Optional  # noqa: E402

from models.model_test import ModelTest  # noqa: E402

from gptqmodel import BACKEND, GPTQModel, QuantizeConfig  # noqa: E402
from gptqmodel.adapter.adapter import Lora  # noqa: E402
from gptqmodel.utils.torch import torch_empty_cache  # noqa: E402
from tests.eval import evaluate, format_eval_result_table  # noqa: E402


# --------Eval METHOD.GPTQ Result---------
# |    Tasks    |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
# |-------------|------:|------|-----:|--------|---|-----:|---|-----:|
# |arc_challenge|      1|none  |     0|acc     |↑  |0.3131|±  |0.0136|
# |             |       |none  |     0|acc_norm|↑  |0.3473|±  |0.0139|
#
# --------Eval METHOD.GPTQ + EoRA Result---------
# |    Tasks    |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
# |-------------|------:|------|-----:|--------|---|-----:|---|-----:|
# |arc_challenge|      1|none  |     0|acc     |↑  |0.3140|±  |0.0136|
# |             |       |none  |     0|acc_norm|↑  |0.3567|±  |0.0140|
class TestQuantAndEORA(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct"  # "meta-llama/Llama-3.2-1B-Instruct"
    DATASET_SIZE_FAST = 32

    EVAL_TASKS = {
        "arc_challenge": {
            "chat_template": True,
            "acc": {"value": 0.3183, "floor_pct": 0.05},
            "acc_norm": {"value": 0.3404, "floor_pct": 0.05},
        },
    }

    QUANT_BATCH_SIZE = 4

    # V2 = False
    # DEBUG = True
    # ACT_GROUP_AWARE = True
    # DESC_ACT = False
    # DATASET_SIZE = 512
    # DATASET_SORT = "desc"
    # USE_FLASH_ATTN = True

    @classmethod
    def setUpClass(cls):
        pass

    @parameterized.expand(
        [
            (METHOD.GPTQ, FORMAT.GPTQ),  # gptq v1
        ]
    )
    def test_quant_and_eora(self, quant_method: METHOD, format: FORMAT):
        rank = 128
        calibration_dataset_concat_size = 0  # disable
        adapter_path = "eora"
        calibration_dataset_size = self._mode_specific_test_setting("DATASET_SIZE")

        with tempfile.TemporaryDirectory() as tmpdir:
            eora = Lora(
                # for quant, path is save path. for load, it is loading path
                path=os.path.join(tmpdir, adapter_path),
                rank=rank,
            )

            quant_config = QuantizeConfig(
                bits=self.BITS,
                group_size=self.GROUP_SIZE,
                desc_act=self.DESC_ACT,  # bitblas only supports DESC_ACT=False
                act_group_aware=self.ACT_GROUP_AWARE,
                adapter=eora,
                format=format,
                quant_method=quant_method,
            )

            model = GPTQModel.load(
                model_id_or_path=self.NATIVE_MODEL_ID,
                quantize_config=quant_config,
                # apply_chat_template=True,
            )

            calibration_dataset = self.load_dataset(model.tokenizer, calibration_dataset_size)

            model.quantize(
                calibration=calibration_dataset,
                # calibration_sort="desc",
                batch_size=self.QUANT_BATCH_SIZE,
                calibration_concat_size=calibration_dataset_concat_size,
            )  #

            # EoRA adapter is saved according to Lora.path property
            # if Lora.path is not set, we will save the lora as "lora.safetensors" in the same path as quant model
            # You can also pass `eora_path` to `model.save()` to override this save path
            model.save(tmpdir)

            del model
            torch_empty_cache()

            for backend in [BACKEND.MARLIN]:  # BACKEND.TORCH_FUSED, BACKEND.BITBLAS, BACKEND.EXLLAMA_V2V BACKEND.MARLIN
                base_bench = self.bench(path=tmpdir, backend=backend, adapter=None)  # inference using qweights only
                eora_bench = self.bench(path=tmpdir, backend=backend, adapter=eora)  # inference using eora (lora)

                print('--------GPT-QModel + EoRA Config ---------')

                # table_data = [[key, value] for key, value in config_dict.items()]
                # print(render_table(table_data, headers=["Key", "Value"], tablefmt="grid"))

                print(f'--------Eval {quant_method} Result---------')
                print(format_eval_result_table(base_bench))

                print(f'--------Eval {quant_method} + EoRA Result---------')
                print(format_eval_result_table(eora_bench))

    def bench(self, path: str, backend: BACKEND, adapter: Optional[Lora]):
        # test post-quant inference
        model = GPTQModel.load(
            model_id_or_path=path,
            backend=backend,
            adapter=adapter,
        )

        tokens = model.generate("Capital of France is")[0]
        result = model.tokenizer.decode(tokens)
        print(f"BACKEND: {backend}, Result: {result}")
        # assert "paris" in result.lower(), f"`paris` not found in `{result}`"

        bench_result = evaluate(
            model_or_id_or_path=model,
            tasks=["arc_challenge"],
            apply_chat_template=True,
            # MMLU is too slow for ci test
            # "mmlu_stem"
        )

        del model
        torch_empty_cache()

        return bench_result
