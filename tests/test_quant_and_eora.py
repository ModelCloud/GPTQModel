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
from gptqmodel import BACKEND, GPTQModel, QuantizeConfig  # noqa: E402
from gptqmodel.adapter.adapter import Lora  # noqa: E402
from gptqmodel.utils.eval import EVAL  # noqa: E402
from gptqmodel.utils.torch import torch_empty_cache  # noqa: E402
from lm_eval.utils import make_table  # noqa: E402
from models.model_test import ModelTest  # noqa: E402
from tabulate import tabulate  # noqa: E402


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
    if "paris" not in result.lower():
        raise AssertionError(" `paris` not found in `result`")

    bench_result = GPTQModel.eval(
        model_or_path=model,
        framework=EVAL.LM_EVAL,
        tasks=[EVAL.LM_EVAL.ARC_CHALLENGE]
    )

    del model
    torch_empty_cache()

    return bench_result

class Test(ModelTest):
    #NATIVE_MODEL_ID = "/monster/data/model/Qwen2.5-0.5B-Instruct/"
    NATIVE_MODEL_ID = "/monster/data/model/Llama-3.2-1B"

    NATIVE_ARC_CHALLENGE_ACC = 0.3567
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.3805
    QUANT_ARC_MAX_DELTA_FLOOR_PERCENT = 0.36

    @classmethod
    def setUpClass(cls):
        pass

    def test_quant_and_eora(self):
        bits = 4
        group_size = 128
        desc_act = True
        rank = 128
        batch_size = 1
        calibration_dataset_rows = 1024
        calibration_dataset_concat_size = 0 # disable
        auto_gc = False
        adapter_file_name = "eora.safetensors"

        config_dict = {
            "bits": bits,
            "group_size": group_size,
            "desc_act": desc_act,
            "rank": rank,
            "batch_size": batch_size,
            "calibration_dataset_rows": calibration_dataset_rows,
            "calibration_dataset_concat_size": calibration_dataset_concat_size,
            "auto_gc": auto_gc,
            "adapter_file_name": adapter_file_name,
        }

        calibration_dataset = load_dataset(
            "allenai/c4",
            data_files="en/c4-train.00001-of-01024.json.gz",
            split="train"
        ).select(range(calibration_dataset_rows))["text"]

        with tempfile.TemporaryDirectory() as tmpdir:
            eora = Lora(
                # for quant, path is save path. for load, it is loading path
                path=os.path.join(tmpdir, adapter_file_name),
                rank=rank,
            )

            quant_config = QuantizeConfig(
                bits=bits,
                group_size=group_size,
                desc_act=desc_act,  # bitblas only supports DESC_ACT=False
                adapter=eora
            )

            model = GPTQModel.load(
                model_id_or_path=self.NATIVE_MODEL_ID,
                quantize_config=quant_config)

            model.quantize(calibration_dataset, batch_size=batch_size, auto_gc=auto_gc, calibration_dataset_concat_size=calibration_dataset_concat_size) #

            # EoRA adapter is saved according to Lora.path property
            # if Lora.path is not set, we will save the lora as "lora.safetensors" in the same path as quant model
            # You can also pass `eora_path` to `model.save()` to override this save path
            model.save(tmpdir)

            del model
            torch_empty_cache()

            # BACKEND.EXLLAMA_V2, BACKEND.EXLLAMA_V1, BACKEND.TRITON, BACKEND.CUDA,
            for backend in [ BACKEND.TORCH ]: # BACKEND.IPEX, BACKEND.BITBLAS, BACKEND.EXLLAMA_V2V BACKEND.MARLIN
                base_bench = bench(path=tmpdir, backend=backend, adapter=None) # inference using qweights only
                eora_bench = bench(path=tmpdir, backend=backend, adapter=eora) # inference using eora (lora)

                print('--------Quant/EoRA Config ---------')

                # Convert the dictionary to a list of lists for tabulate
                table_data = [[key, value] for key, value in config_dict.items()]
                print(tabulate(table_data, headers=["Key", "Value"], tablefmt="grid"))

                print('--------Eval Base Result---------')
                print(make_table(base_bench))
                if "groups" in base_bench:
                    print(make_table(base_bench, "groups"))

                print('--------Eval EoRA Result---------')
                print(make_table(eora_bench))
                if "groups" in eora_bench:
                    print(make_table(eora_bench, "groups"))
