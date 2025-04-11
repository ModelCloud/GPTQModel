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

import torch
from peft.tuners.lora.gptq import GPTQLoraLinear
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

import tempfile  # noqa: E402
from typing import Optional  # noqa: E402

from datasets import load_dataset  # noqa: E402
from gptqmodel import BACKEND, GPTQModel, QuantizeConfig  # noqa: E402
from gptqmodel.adapter.adapter import HF_ADAPTER_FILE_NAME, HF_ADAPTER_WEIGHT_KEY_PREFIX, Lora  # noqa: E402
from gptqmodel.utils.eval import EVAL  # noqa: E402
from gptqmodel.utils.torch import torch_empty_cache  # noqa: E402
from lm_eval.utils import make_table  # noqa: E402
from logbar import LogBar
from models.model_test import ModelTest  # noqa: E402
from tabulate import tabulate  # noqa: E402

log = LogBar.shared()


class Test(ModelTest):
    # NATIVE_MODEL_ID = "/monster/data/model/Qwen2.5-0.5B-Instruct/"
    # NATIVE_MODEL_ID = "/monster/data/model/tinyllama-15M-stories"
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
        calibration_dataset_rows = 512
        calibration_dataset_concat_size = 0  # disable
        auto_gc = False
        adapter_path = "eora"
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
            "auto_gc": auto_gc,
            "adapter_path": adapter_path,
        }

        calibration_dataset = load_dataset(
            dataset_id,
            data_files=dataset_files,
            split="train"
        ).select(range(calibration_dataset_rows))["text"]

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
                adapter=eora,
                dynamic={
                    ".*\\.gate_proj.*": {
                        "adapter": {
                            "rank": 256
                        }
                    }
                },
            )

            model = GPTQModel.load(
                model_id_or_path=self.NATIVE_MODEL_ID,
                quantize_config=quant_config,
            )

            model.quantize(
                calibration_dataset=calibration_dataset,
                batch_size=batch_size,
                auto_gc=auto_gc,
                calibration_dataset_concat_size=calibration_dataset_concat_size,
            )  #

            # EoRA adapter is saved according to Lora.path property
            # if Lora.path is not set, we will save the lora as "lora.safetensors" in the same path as quant model
            # You can also pass `eora_path` to `model.save()` to override this save path
            model.save(tmpdir)

            del model
            torch_empty_cache()

            # BACKEND.EXLLAMA_V2, BACKEND.EXLLAMA_V1, BACKEND.TRITON, BACKEND.CUDA,
            for backend in [BACKEND.MARLIN]:  # BACKEND.IPEX, BACKEND.BITBLAS, BACKEND.EXLLAMA_V2V BACKEND.MARLIN
                eora_bench = self.bench(path=tmpdir, backend=backend, adapter=eora)  # inference using eora (lora)
                base_bench = self.bench(path=tmpdir, backend=backend, adapter=None)  # inference using qweights only

                print('--------GPTQModel + EoRA Config ---------')

                # Convert the dictionary to a list of lists for tabulate
                table_data = [[key, value] for key, value in config_dict.items()]
                print(tabulate(table_data, headers=["Key", "Value"], tablefmt="grid"))

                print('--------Eval GPTQ Result---------')
                print(make_table(base_bench))
                if "groups" in base_bench:
                    print(make_table(base_bench, "groups"))

                print('--------Eval GPTQ + EoRA Result---------')
                print(make_table(eora_bench))
                if "groups" in eora_bench:
                    print(make_table(eora_bench, "groups"))

    def bench(self, path: str, backend: BACKEND, adapter: Optional[Lora]):
        # test post-quant inference
        if adapter:
            adapter_weights = load_file(os.path.join(adapter.path, HF_ADAPTER_FILE_NAME))
            origin_lora_a_weight = adapter_weights[
                f"{HF_ADAPTER_WEIGHT_KEY_PREFIX}model.layers.5.self_attn.v_proj.lora_A.weight"]
            origin_lora_b_weight = adapter_weights[
                f"{HF_ADAPTER_WEIGHT_KEY_PREFIX}model.layers.5.self_attn.v_proj.lora_B.weight"]

            model = AutoModelForCausalLM.from_pretrained(path, device_map="cuda")
            log.info("PEFT: converting model to lora model")
            model.load_adapter(adapter.path)

            self.assert_adapter_load(model, origin_lora_a_weight, origin_lora_b_weight)
            del model

            model = AutoModelForCausalLM.from_pretrained(adapter.path, device_map="cuda")
            log.info("PEFT: load model by adapter.path")

            self.assert_adapter_load(model, origin_lora_a_weight, origin_lora_b_weight)
            print("peft model", model)

            # assert dynamic rank
            v_proj_module = model.model.layers[5].self_attn.v_proj
            assert v_proj_module.lora_A["default"].weight.data.shape[0] == 128
            assert v_proj_module.lora_B["default"].weight.data.shape[1] == 128
            gate_proj_module = model.model.layers[5].mlp.gate_proj
            assert gate_proj_module.lora_A["default"].weight.data.shape[0] == 256
            assert gate_proj_module.lora_B["default"].weight.data.shape[1] == 256

            del origin_lora_a_weight, origin_lora_b_weight, adapter_weights
        else:
            model = AutoModelForCausalLM.from_pretrained(path, device_map="cuda")
            print("model", model)

        tokenizer = AutoTokenizer.from_pretrained(path)
        inp = tokenizer("Capital of France is", return_tensors="pt").to(model.device)
        tokens = model.generate(**inp)[0]
        result = tokenizer.decode(tokens)
        print(f"BACKEND: {backend}, Result: {result}")
        # assert "paris" in result.lower(), f"`paris` not found in `{result}`"

        bench_result = GPTQModel.eval(
            model_or_id_or_path=model,
            framework=EVAL.LM_EVAL,
            tasks=[EVAL.LM_EVAL.ARC_CHALLENGE, EVAL.LM_EVAL.MMLU],
        )

        del model
        torch_empty_cache()

        return bench_result

    def assert_adapter_load(self, model, origin_lora_a_weight, origin_lora_b_weight):
        module = model.model.layers[5].self_attn.v_proj
        assert isinstance(module, GPTQLoraLinear)
        assert torch.equal(origin_lora_a_weight.to(model.device), module.lora_A["default"].weight.data)
        assert torch.equal(origin_lora_b_weight.to(model.device), module.lora_B["default"].weight.data)
