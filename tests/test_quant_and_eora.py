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
import tempfile

from datasets import load_dataset

from gptqmodel.utils.eval import EVAL
from gptqmodel.utils.torch import torch_empty_cache

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

from gptqmodel import BACKEND, GPTQModel, QuantizeConfig  # noqa: E402
from gptqmodel.adapter.adapter import Lora  # noqa: E402
from models.model_test import ModelTest  # noqa: E402


class Test(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Qwen2.5-0.5B-Instruct/"

    NATIVE_ARC_CHALLENGE_ACC = 0.3567
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.3805
    QUANT_ARC_MAX_DELTA_FLOOR_PERCENT = 0.36

    @classmethod
    def setUpClass(cls):
        pass

    def test_quant_and_eora(self):
        calibration_dataset = load_dataset(
            "allenai/c4",
            data_files="en/c4-train.00001-of-01024.json.gz",
            split="train"
        ).select(range(128))["text"]

        with tempfile.TemporaryDirectory() as tmpdir:
            quant_config = QuantizeConfig(
                bits=4,
                group_size=32,
                desc_act=False,  # bitblas only supports DESC_ACT=False
                adapter=Lora(
                    path=os.path.join(tmpdir, "lora_adapter.safetensors"),
                    rank=512,
                )
            )

            model = GPTQModel.load(self.NATIVE_MODEL_ID, quant_config)

            # increase `batch_size` to match gpu/vram specs to speed up quantization
            model.quantize(calibration_dataset, batch_size=1, auto_gc=False)
            # print("log", l)
            # model.quantize_old(calibration_dataset, batch_size=2)

            model.save(tmpdir)
            # .reshape(out_shape)
            for backend in [ BACKEND.TORCH,
                            ]: # BACKEND.IPEX, BACKEND.BITBLAS, BACKEND.EXLLAMA_V2V BACKEND.MARLIN
                # test post-quant inference
                model = GPTQModel.load(
                    model_id_or_path=tmpdir,
                    backend=backend,
                )
                tokens = model.generate("Capital of France is")[0]
                result = model.tokenizer.decode(tokens)
                print(f"BACKEND: {backend}, Result: {result}")
                self.assertIn("paris", result.lower())

                GPTQModel.eval(
                    model_or_path=model,
                    #backend=BACKEND.EXLLAMA_V2,
                    framework=EVAL.LM_EVAL,
                    tasks=[EVAL.LM_EVAL.ARC_CHALLENGE]
                )

                del model
                torch_empty_cache()
