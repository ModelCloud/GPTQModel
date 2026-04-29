# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# -- do not touch
import os


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch
import logging  # noqa: E402
import tempfile  # noqa: E402
import traceback  # noqa: E402
import unittest  # noqa: E402

from transformers import AutoTokenizer  # noqa: E402

from gptqmodel import BACKEND, GPTQModel, QuantizeConfig  # noqa: E402
from gptqmodel.nn_modules.qlinear.bitblas import BitBLASLinear  # noqa: E402
from gptqmodel.nn_modules.qlinear.exllamav2 import ExllamaV2Linear  # noqa: E402
from gptqmodel.nn_modules.qlinear.marlin import MarlinLinear  # noqa: E402
from gptqmodel.nn_modules.qlinear.torch import TorchLinear  # noqa: E402
from gptqmodel.nn_modules.qlinear.tritonv2 import TritonV2Linear  # noqa: E402
from tests.eval import evaluate, format_eval_result_table, get_eval_task_metrics  # noqa: E402


logger = logging.getLogger(__name__)

RAND_SEED = 42
TASK_NAME = "arc_challenge"

class TestGroupSize(unittest.TestCase):
    QLINEAR_DICT = {
        BACKEND.EXLLAMA_V2: ExllamaV2Linear,
        BACKEND.TRITON: TritonV2Linear,
        BACKEND.TORCH: TorchLinear,
        BACKEND.BITBLAS: BitBLASLinear,
        BACKEND.MARLIN: MarlinLinear,
    }


    @classmethod
    def setUpClass(cls):
        cls.pack_backends = [BACKEND.TRITON, BACKEND.TORCH, BACKEND.BITBLAS]
        cls.backends = list(cls.pack_backends)
        cls.backends.extend([BACKEND.EXLLAMA_V2, BACKEND.MARLIN, ])

    def test_group_size(self):
        # quantize
        OPT_MODEL_ID = "/monster/data/model/opt-125m"
        model_id = OPT_MODEL_ID
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        dataset = [
            "gptqmodel is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."]
        calibration_dataset = [tokenizer(example) for example in dataset]
        for quant_backend in self.pack_backends:
            group_sizes = self.QLINEAR_DICT[quant_backend].SUPPORTS_GROUP_SIZE
            for group_size in group_sizes:
                print("-----------------------quant-----------------------")
                quantize_config = QuantizeConfig(bits=4, group_size=group_size, sym=True, desc_act=False)
                print(f"group_size: {quantize_config.group_size}, quant_backend: {quant_backend} start quant")
                try:
                    self.quant_and_eval(calibration_dataset, model_id, quant_backend, quantize_config, tokenizer)
                except Exception:
                    print(f"{quantize_config.group_size}, quant_backend: {quant_backend} An error occurred")
                    traceback.print_exc()
                    continue

    def quant_and_eval(self, calibration_dataset, model_id, quant_backend, quantize_config, tokenizer):
        model = GPTQModel.load(
            model_id,
            quantize_config=quantize_config,
        )
        model.quantize(calibration_dataset, backend=quant_backend)
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save(
                tmp_dir,
            )
            tokenizer.save_pretrained(tmp_dir)

            del model

            for inference_backend in self.backends:
                if quantize_config.group_size not in self.QLINEAR_DICT[inference_backend].SUPPORTS_GROUP_SIZE:
                    # Skip inference_backend that does not support the current group_size
                    continue

                try:
                    self.eval(inference_backend, quant_backend, quantize_config, tmp_dir)
                except Exception:
                    traceback.print_exc()
                    continue

    def eval(self, inference_backend, quant_backend, quantize_config, tmp_dir):
        print("-----------------------eval-----------------------")
        print(
            f'{quantize_config.group_size}, quant_backend: {quant_backend}, inference_backend: {inference_backend}. start eval')
        model = GPTQModel.load(
            tmp_dir,
            device_map="auto",
            backend=inference_backend,
        )
        results = evaluate(
            model_or_id_or_path=model,
            output_path=tmp_dir,
            tasks=TASK_NAME,
            apply_chat_template=False,
            trust_remote_code=False,
            batch_size=32,
            gen_kwargs="temperature=0.0,top_k=50",
        )
        print('--------Eval Result---------')
        print(format_eval_result_table(results))
        print('--------Eval Result End---------')
        task_results = {
            metric: value for metric, value in get_eval_task_metrics(results, TASK_NAME).items()
            if metric != 'alias' and 'stderr' not in metric
        }
        print(
            f"group_size is: {quantize_config.group_size}, quant_backend: {quant_backend}, inference_backend: {inference_backend} -> task_results: {task_results}")
        del model
