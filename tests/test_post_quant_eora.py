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
from pathlib import Path  # noqa: E402
from typing import Optional  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402
from datasets import load_dataset
from models.model_test import ModelTest  # noqa: E402

from gptqmodel import BACKEND, CalibrationConfig, GPTQModel  # noqa: E402
from gptqmodel.adapter.adapter import Lora  # noqa: E402
from gptqmodel.utils.torch import torch_empty_cache  # noqa: E402
from tests.eval import evaluate  # noqa: E402


pytestmark = [pytest.mark.model, pytest.mark.slow]

# Local C4 shard shared with the other calibration tests; offline runs cannot
# resolve the Hub `allenai/c4` builder.
C4_LOCAL = "/monster/data/model/dataset/c4-train.00000-of-01024.json.gz"


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

    result = ModelTest.generate_stable_with_limit(
        model,
        model.tokenizer,
        "The capital city of France is named",
        max_new_tokens=128,
    )
    print(f"BACKEND: {backend}, Result: {result}")
    if "paris" not in result.lower():
        raise AssertionError(" `paris` not found in `result`")

    bench_result = evaluate(
        model_or_id_or_path=model,
        tasks=["arc_challenge", "mmlu_stem"]
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

    # 512 full C4 rows x 16 Llama-3.2-1B layers at rank 256: on a CPU host the
    # run had finished only layer 0's first subset after 29 minutes.
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="EoRA generation at this scale needs CUDA")
    def test_post_quant_eora(self):
        for required in (self.NATIVE_MODEL_ID, self.QUANTIZED_MODEL_PATH):
            if not Path(required).is_dir():
                pytest.skip(f"local checkpoint missing: {required}")
        if not Path(C4_LOCAL).is_file():
            pytest.skip(f"local C4 calibration shard missing: {C4_LOCAL}")

        rank = 256
        calibration_dataset_rows = 512
        calibration_dataset_concat_size = None  # disable

        calibration_dataset = load_dataset(
            "json",
            data_files=C4_LOCAL,
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
                calibration=CalibrationConfig(
                    dataset=calibration_dataset,
                    concat_size=calibration_dataset_concat_size,
                ),
            )

            # for backend in [BACKEND.MARLIN]:  # BACKEND.TORCH_FUSED, BACKEND.BITBLAS, BACKEND.EXLLAMA_V2V BACKEND.MARLIN
            #     base_bench = bench(path=self.QUANTIZED_MODEL_PATH, backend=backend, adapter=None)  # inference using qweights only
            #     eora_bench = bench(path=self.QUANTIZED_MODEL_PATH, backend=backend, adapter=eora)  # inference using eora (lora)
            #
            #     print('--------Quant/EoRA Config ---------')
            #
            #     # Render the config dictionary as a two-column table
            #     table_data = [[key, value] for key, value in config_dict.items()]
            #     print(render_table(table_data, headers=["Key", "Value"], tablefmt="grid"))
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
