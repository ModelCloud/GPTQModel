# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# -- do not touch
import os


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

import torch
from parameterized import parameterized  # noqa: E402
from peft.tuners.lora.gptq import GPTQLoraLinear
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from gptqmodel.quantization import FORMAT, METHOD  # noqa: E402


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

import tempfile  # noqa: E402
from typing import Optional  # noqa: E402

from datasets import load_dataset  # noqa: E402
from lm_eval.utils import make_table  # noqa: E402
from logbar import LogBar
from models.model_test import ModelTest  # noqa: E402
from tabulate import tabulate  # noqa: E402

from gptqmodel import BACKEND, GPTQModel, QuantizeConfig  # noqa: E402
from gptqmodel.adapter.adapter import HF_ADAPTER_FILE_NAME, HF_ADAPTER_WEIGHT_KEY_PREFIX, Lora  # noqa: E402
from gptqmodel.utils.eval import EVAL  # noqa: E402
from gptqmodel.utils.torch import torch_empty_cache  # noqa: E402


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

    EVAL_TASKS = {
        EVAL.LM_EVAL.ARC_CHALLENGE: {
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

            calibration_dataset = self.load_dataset(model.tokenizer, self.DATASET_SIZE)

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

            # BACKEND.EXLLAMA_V2, BACKEND.EXLLAMA_V1, BACKEND.TRITON, BACKEND.CUDA,
            for backend in [BACKEND.MARLIN]:  # BACKEND.IPEX, BACKEND.BITBLAS, BACKEND.EXLLAMA_V2V BACKEND.MARLIN
                base_bench = self.bench(path=tmpdir, backend=backend, adapter=None)  # inference using qweights only
                eora_bench = self.bench(path=tmpdir, backend=backend, adapter=eora)  # inference using eora (lora)

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
        # assert "paris" in result.lower(), f"`paris` not found in `{result}`"

        bench_result = GPTQModel.eval(
            model_or_id_or_path=model,
            framework=EVAL.LM_EVAL,
            tasks=[EVAL.LM_EVAL.ARC_CHALLENGE],
            apply_chat_template=True,
            # MMLU is too slow for ci test
            # EVAL.LM_EVAL.MMLU_STEM
        )

        del model
        torch_empty_cache()

        return bench_result


######## test_quant_and_eora_transformers.py #########


log = LogBar.shared()


class TestTransformers(ModelTest):
    # NATIVE_MODEL_ID = "/monster/data/model/Qwen2.5-0.5B-Instruct/"
    # NATIVE_MODEL_ID = "/monster/data/model/tinyllama-15M-stories"
    NATIVE_MODEL_ID = "/monster/data/model/Llama-3.2-1B"

    EVAL_TASKS = {
        EVAL.LM_EVAL.ARC_CHALLENGE: {
            "acc": {"value": 0.3567, "floor_pct": 0.36},
            "acc_norm": {"value": 0.3805, "floor_pct": 0.36},
        },
    }

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
                calibration=calibration_dataset,
                batch_size=batch_size,
                calibration_concat_size=calibration_dataset_concat_size,
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
            tasks=[EVAL.LM_EVAL.ARC_CHALLENGE, EVAL.LM_EVAL.MMLU_STEM],
        )

        del model
        torch_empty_cache()

        return bench_result

    def assert_adapter_load(self, model, origin_lora_a_weight, origin_lora_b_weight):
        module = model.model.layers[5].self_attn.v_proj
        assert isinstance(module, GPTQLoraLinear)
        assert torch.equal(origin_lora_a_weight.to(model.device), module.lora_A["default"].weight.data)
        assert torch.equal(origin_lora_b_weight.to(model.device), module.lora_B["default"].weight.data)
