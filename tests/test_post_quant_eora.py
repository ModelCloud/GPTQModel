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

from datasets import load_dataset
from models.model_test import ModelTest  # noqa: E402

from gptqmodel import BACKEND, GPTQModel  # noqa: E402
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
    if "paris" not in result.lower():
        raise AssertionError(" `paris` not found in `result`")

    bench_result = GPTQModel.eval(
        model_or_id_or_path=model,
        framework=EVAL.LM_EVAL,
        tasks=[EVAL.LM_EVAL.ARC_CHALLENGE, EVAL.LM_EVAL.MMLU_STEM]
    )

    del model
    torch_empty_cache()

    return bench_result


class TestEoraPostQuant(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct"
    QUANTIZED_MODEL_PATH = "/monster/data/model/Llama-3.2-1B-Instruct-gptqmodel-4bit-vortex-v1/"

    @classmethod
    def setUpClass(cls):
        pass

    def test_post_quant_eora(self):
        rank = 256
        calibration_dataset_rows = 512
        calibration_dataset_concat_size = 0  # disable

        dataset_id = "allenai/c4"
        dataset_files = "en/c4-train.00001-of-01024.json.gz"

        calibration_dataset = load_dataset(
            dataset_id,
            data_files=dataset_files,
            split="train"
        ).select(range(calibration_dataset_rows))["text"]

        with tempfile.TemporaryDirectory() as tmpdir:
            eora = Lora(
                # for eora generation, path is adapter save path; for load, it is loading path
                path=os.path.join(tmpdir),
                rank=rank,
            )

            # eora generation and save in one step
            GPTQModel.adapter.generate(
                adapter=eora,
                model_id_or_path=self.NATIVE_MODEL_ID,
                quantized_model_id_or_path=self.QUANTIZED_MODEL_PATH,
                calibration_dataset=calibration_dataset,
                calibration_dataset_concat_size=calibration_dataset_concat_size,
            )

            # BACKEND.EXLLAMA_V2, BACKEND.EXLLAMA_V1, BACKEND.TRITON, BACKEND.CUDA,
            # for backend in [BACKEND.MARLIN]:  # BACKEND.IPEX, BACKEND.BITBLAS, BACKEND.EXLLAMA_V2V BACKEND.MARLIN
            #     base_bench = bench(path=self.QUANTIZED_MODEL_PATH, backend=backend, adapter=None)  # inference using qweights only
            #     eora_bench = bench(path=self.QUANTIZED_MODEL_PATH, backend=backend, adapter=eora)  # inference using eora (lora)
            #
            #     print('--------Quant/EoRA Config ---------')
            #
            #     # Convert the dictionary to a list of lists for tabulate
            #     table_data = [[key, value] for key, value in config_dict.items()]
            #     print(tabulate(table_data, headers=["Key", "Value"], tablefmt="grid"))
            #
            #     print('--------Eval Base Result---------')
            #     print(make_table(base_bench))
            #     if "groups" in base_bench:
            #         print(make_table(base_bench, "groups"))
            #
            #     print('--------Eval EoRA Result---------')
            #     print(make_table(eora_bench))
            #     if "groups" in eora_bench:
            #         print(make_table(eora_bench, "groups"))
