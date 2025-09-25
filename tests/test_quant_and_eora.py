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
from gptqmodel.quantization import FORMAT, METHOD  # noqa: E402
from gptqmodel.utils.eval import EVAL  # noqa: E402
from gptqmodel.utils.torch import torch_empty_cache  # noqa: E402
from lm_eval.utils import make_table  # noqa: E402
from models.model_test import ModelTest  # noqa: E402
from parameterized import parameterized  # noqa: E402


class Test(ModelTest):
    #NATIVE_MODEL_ID = "/monster/data/model/Qwen2.5-0.5B-Instruct/"
    #NATIVE_MODEL_ID = "/monster/data/model/tinyllama-15M-stories"
    NATIVE_MODEL_ID = "/monster/data/model/Llama-3.2-1B"

    NATIVE_ARC_CHALLENGE_ACC = 0.3567
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.3805
    QUANT_ARC_MAX_DELTA_FLOOR_PERCENT = 0.36

    @classmethod
    def setUpClass(cls):
        pass

    @parameterized.expand(
        [
            # (QUANT_METHOD.GPTQ, FORMAT.GPTQ, True), # gptq v2
            (METHOD.GPTQ, FORMAT.GPTQ, False), # gptq v1
            #(QUANT_METHOD.QQQ, FORMAT.QQQ),
        ]
    )
    def test_quant_and_eora(self, quant_method: METHOD, format: FORMAT, v2: bool):
        bits = 4
        group_size = 128
        desc_act = False
        act_group_aware = True
        rank = 128
        batch_size = 1
        calibration_dataset_rows = 512
        calibration_dataset_concat_size = 0 # disable
        auto_gc = False
        adapter_path = "eora"
        dataset_id = "allenai/c4"
        dataset_files = "en/c4-train.00001-of-01024.json.gz"


        calibration_dataset = load_dataset(
            dataset_id,
            data_files=dataset_files,
            split="train"
        ).select(range(calibration_dataset_rows))["text"]

        # with gzip.open("/monster/data/model/dataset/c4-train.00000-of-01024.json.gz", 'rt', encoding='utf-8') as f:
        #     data = [json.loads(line)["text"] for line in f]
        #     calibration_dataset = data[:calibration_dataset_rows]

        with tempfile.TemporaryDirectory() as tmpdir:
            eora = Lora(
                # for quant, path is save path. for load, it is loading path
                path=os.path.join(tmpdir, adapter_path),
                rank=rank,
            )

            quant_config = QuantizeConfig(
                bits=bits,
                group_size=group_size,
                desc_act=desc_act,  # bitblas only supports DESC_ACT=False
                act_group_aware=act_group_aware,
                adapter=eora,
                format=format,
                quant_method=quant_method,
                v2=v2,
            )

            model = GPTQModel.load(
                model_id_or_path=self.NATIVE_MODEL_ID,
                quantize_config=quant_config,
            )

            model.quantize(
                calibration=calibration_dataset,
                batch_size=batch_size,
                auto_gc=auto_gc,
                calibration_concat_size=calibration_dataset_concat_size,
            ) #

            # EoRA adapter is saved according to Lora.path property
            # if Lora.path is not set, we will save the lora as "lora.safetensors" in the same path as quant model
            # You can also pass `eora_path` to `model.save()` to override this save path
            model.save(tmpdir)

            del model
            torch_empty_cache()

            # BACKEND.EXLLAMA_V2, BACKEND.EXLLAMA_V1, BACKEND.TRITON, BACKEND.CUDA,
            for backend in [ BACKEND.AUTO ]: # BACKEND.IPEX, BACKEND.BITBLAS, BACKEND.EXLLAMA_V2V BACKEND.MARLIN
                base_bench = self.bench(path=tmpdir, backend=backend, adapter=None) # inference using qweights only
                eora_bench = self.bench(path=tmpdir, backend=backend, adapter=eora) # inference using eora (lora)

                print('--------GPTQModel + EoRA Config ---------')

                # Convert the dictionary to a list of lists for tabulate
                # table_data = [[key, value] for key, value in config_dict.items()]
                # print(tabulate(table_data, headers=["Key", "Value"], tablefmt="grid"))

                print(f'--------Eval {quant_method} Result---------')
                print(make_table(base_bench))
                if "groups" in base_bench:
                    print(make_table(base_bench, "groups"))

                print(f'--------Eval {quant_method} + EoRA Result---------')
                print(make_table(eora_bench))
                if "groups" in eora_bench:
                    print(make_table(eora_bench, "groups"))

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
        #assert "paris" in result.lower(), f"`paris` not found in `{result}`"

        bench_result = GPTQModel.eval(
            model_or_id_or_path=model,
            framework=EVAL.LM_EVAL,
            tasks=[EVAL.LM_EVAL.ARC_CHALLENGE],
            # MMLU is too slow for ci test
            # EVAL.LM_EVAL.MMLU
        )

        del model
        torch_empty_cache()

        return bench_result
