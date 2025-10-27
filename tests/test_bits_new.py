# Copyright 2025 ModelCloud
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

import tempfile  # noqa: E402
from typing import Optional  # noqa: E402

from datasets import load_dataset  # noqa: E402
from lm_eval.utils import make_table  # noqa: E402
from models.model_test import ModelTest  # noqa: E402
from tabulate import tabulate  # noqa: E402

from gptqmodel import BACKEND, GPTQModel, QuantizeConfig  # noqa: E402
from gptqmodel.adapter.adapter import Lora  # noqa: E402
from gptqmodel.utils.eval import EVAL  # noqa: E402
from gptqmodel.utils.torch import torch_empty_cache  # noqa: E402


def bench(path: str, backend: BACKEND, adapter: Optional[Lora]):
    # test post-quant inference
    model = GPTQModel.load(
        model_id_or_path=path,
        backend=backend,
        adapter=adapter,
    )

    # torch can benefit from optimization
    if backend == BACKEND.TORCH:
        model.optimize()

    tokens = model.generate("Capital of France is")[0]
    result = model.tokenizer.decode(tokens)
    print(f"BACKEND: {backend}, Result: {result}")
    # assert "paris" in result.lower(), f"`paris` not found in `{result}`"

    bench_result = GPTQModel.eval(
        model_or_id_or_path=model,
        framework=EVAL.LM_EVAL,
        tasks=[EVAL.LM_EVAL.ARC_CHALLENGE, EVAL.LM_EVAL.MMLU_STEM],
        batch_size=16,
    )

    del model
    torch_empty_cache()

    return bench_result

class Test(ModelTest):
    # NATIVE_MODEL_ID = "/monster/data/model/Qwen2.5-0.5B-Instruct/"
    #NATIVE_MODEL_ID = "/monster/data/model/tinyllama-15M-stories"
    # NATIVE_MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct"
    # NATIVE_MODEL_ID = "/monster/data/model/Llama-3.2-3B-Instruct"


    EVAL_TASKS = {
        EVAL.LM_EVAL.ARC_CHALLENGE: {
            "acc": {"value": 0.3567, "floor_pct": 0.36},
            "acc_norm": {"value": 0.3805, "floor_pct": 0.36},
        },
    }

    @classmethod
    def setUpClass(cls):
        pass
# clear && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 BITS=2 NATIVE_MODEL_ID=/monster/data/model/Llama-3.2-1B-Instruct pytest tests/test_quant_and_eora.py
# clear && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 BITS=3 NATIVE_MODEL_ID=/monster/data/model/Llama-3.2-1B-Instruct pytest tests/test_quant_and_eora.py
# clear && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 BITS=4 NATIVE_MODEL_ID=/monster/data/model/Llama-3.2-1B-Instruct pytest tests/test_quant_and_eora.py
# clear && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 BITS=8 NATIVE_MODEL_ID=/monster/data/model/Llama-3.2-1B-Instruct pytest tests/test_quant_and_eora.py
#
# clear && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=4 BITS=2 NATIVE_MODEL_ID=/monster/data/model/Llama-3.2-3B-Instruct pytest tests/test_quant_and_eora.py
# clear && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 BITS=3 NATIVE_MODEL_ID=/monster/data/model/Llama-3.2-3B-Instruct pytest tests/test_quant_and_eora.py
# clear && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7 BITS=4 NATIVE_MODEL_ID=/monster/data/model/Llama-3.2-3B-Instruct pytest tests/test_quant_and_eora.py
# clear && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 BITS=8 NATIVE_MODEL_ID=/monster/data/model/Llama-3.2-3B-Instruct pytest tests/test_quant_and_eora.py


    def test_quant_and_eora(self):
        bits = int(os.environ["BITS"])
        self.NATIVE_MODEL_ID = os.environ["NATIVE_MODEL_ID"]

        print(f"eeeeee gpu: testing {bits}: bits, model: {self.NATIVE_MODEL_ID}")
        group_size = 128
        desc_act = True
        rank = 128
        batch_size = 1
        calibration_dataset_rows = 512
        calibration_dataset_concat_size = 0 # disable
        adapter_file_name = "eora.safetensors"
        dataset_id = "allenai/c4"
        dataset_files = "en/c4-train.00001-of-01024.json.gz"

        config_dict = {
            "model_id": self.NATIVE_MODEL_ID,
            "dataset_id": dataset_id,
            "dataset_files": dataset_files,
            "bits": bits,
            "group_size": group_size,
            "desc_act": desc_act,
            "rank": rank,
            "batch_size": batch_size,
            "calibration_dataset_rows": calibration_dataset_rows,
            "calibration_dataset_concat_size": calibration_dataset_concat_size,
            "adapter_file_name": adapter_file_name,
        }

        calibration_dataset = load_dataset(
            dataset_id,
            data_files=dataset_files,
            split="train"
        ).select(range(calibration_dataset_rows))["text"]

        with tempfile.TemporaryDirectory():
            # eora = Lora(
            #     # for quant, path is save path. for load, it is loading path
            #     path=os.path.join(tmpdir, adapter_file_name),
            #     rank=rank,
            # )

            quant_config = QuantizeConfig(
                bits=bits,
                group_size=group_size,
                desc_act=desc_act,  # bitblas only supports DESC_ACT=False
                # adapter=eora,
            )

            save_path=os.path.join(f"./{quant_config.bits}", self.NATIVE_MODEL_ID.removeprefix("/monster/data/model/"))

            if os.path.exists(save_path):
                self.NATIVE_MODEL_ID=save_path

            model = GPTQModel.load(
                model_id_or_path=self.NATIVE_MODEL_ID,
                quantize_config=quant_config,
            )

            if not model.quantized:
                model.quantize(
                    calibration=calibration_dataset,
                    batch_size=batch_size,
                    calibration_concat_size=calibration_dataset_concat_size,
                    backend=BACKEND.TORCH,
                ) #


                # EoRA adapter is saved according to Lora.path property
                # if Lora.path is not set, we will save the lora as "lora.safetensors" in the same path as quant model
                # You can also pass `eora_path` to `model.save()` to override this save path
                model.save(save_path)

                del model
                torch_empty_cache()

            # BACKEND.EXLLAMA_V2, BACKEND.EXLLAMA_V1, BACKEND.TRITON, BACKEND.CUDA,
            for backend in [ BACKEND.TORCH ]: # BACKEND.IPEX, BACKEND.BITBLAS, BACKEND.EXLLAMA_V2V BACKEND.MARLIN
                base_bench = bench(path=save_path, backend=backend, adapter=None) # inference using qweights only
                # eora_bench = bench(path=tmpdir, backend=backend, adapter=eora) # inference using eora (lora)

                print('--------GPTQModel + EoRA Config ---------')

                # Convert the dictionary to a list of lists for tabulate
                table_data = [[key, value] for key, value in config_dict.items()]
                print(tabulate(table_data, headers=["Key", "Value"], tablefmt="grid"))

                print('--------Eval GPTQ Result---------')
                print(make_table(base_bench))
                if "groups" in base_bench:
                    print(make_table(base_bench, "groups"))

                # print('--------Eval GPTQ + EoRA Result---------')
                # print(make_table(eora_bench))
                # if "groups" in eora_bench:
                #     print(make_table(eora_bench, "groups"))
