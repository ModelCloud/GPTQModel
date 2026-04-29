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

class TestBits(unittest.TestCase):
    QLINEAR_DICT = {
        BACKEND.EXLLAMA_V2: ExllamaV2Linear,
        BACKEND.TRITON: TritonV2Linear,
        BACKEND.TORCH: TorchLinear,
        BACKEND.BITBLAS: BitBLASLinear,
        BACKEND.MARLIN: MarlinLinear,
    }

    QUANT_ARC_MAX_DELTA_FLOOR_PERCENT = 0.2
    QUANT_ARC_MAX_POSITIVE_DELTA_CEIL_PERCENT = 0.2

    CUDA_QLINEAR_QUANTIZED_MODEL_ARC_CHALLENGE_EXPECTS = {
        2: {'accuracy,loglikelihood': 0.2150170648464164, 'accuracy,loglikelihood_norm': 0.2696245733788396},
        3: {'accuracy,loglikelihood': 0.2175767918088737, 'accuracy,loglikelihood_norm': 0.26621160409556316},
        4: {'accuracy,loglikelihood': 0.2363, 'accuracy,loglikelihood_norm': 0.2517},
        8: {'accuracy,loglikelihood': 0.3020, 'accuracy,loglikelihood_norm': 0.3319112627986348},
    }

    def calculatorPer(self, filter, value, base_value):
        diff_pct = (value / base_value) * 100
        print(f"{filter}: {value} diff {diff_pct:.2f}%")
        return diff_pct

    def check_results(self, bits: int, task_results):
        for filter, value in task_results.items():
            base_value = self.CUDA_QLINEAR_QUANTIZED_MODEL_ARC_CHALLENGE_EXPECTS[bits][filter]
            diff_pct = self.calculatorPer(filter=filter, value=value, base_value=base_value)
            negative_pct = 100 * (1 - self.QUANT_ARC_MAX_DELTA_FLOOR_PERCENT)
            positive_pct = 100 * (1 + self.QUANT_ARC_MAX_POSITIVE_DELTA_CEIL_PERCENT)
            self.assertTrue(negative_pct <= diff_pct <= positive_pct, f"{filter}: {value} diff {diff_pct:.2f}% is out of the expected range [{negative_pct}-{positive_pct}%], expected: {base_value}")

    @classmethod
    def setUpClass(cls):
        # TODO Only CUDA Quant Linear is tested for now
        cls.pack_backends = [BACKEND.TRITON]
        cls.backends = [BACKEND.MARLIN]

    def test_bits(self):
        # quantize
        model_id = "/monster/data/model/Qwen2.5-0.5B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        calibration_dataset = ["gptqmodel is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."]

        for quant_backend in self.pack_backends:
            supports_bits = self.QLINEAR_DICT[quant_backend].SUPPORTS_BITS
            for bits in supports_bits:
                print(f"-----------------------quant backend: {quant_backend}-- bits: {bits} ---------------------")
                quantize_config = QuantizeConfig(bits=bits, group_size=128, sym=True, desc_act=True)
                print(f"bits: {bits}, quant_backend: {quant_backend} start quant")
                #try:
                self.quant_and_eval(calibration_dataset, model_id, quant_backend, quantize_config, tokenizer)
                # except Exception as e:
                #     raise e
                #     # error_log=f"bits:  {bits}, quant_backend: {quant_backend} An error occurred"
                    # print(error_log)
                    # errors.append(error_log)
                    #
                    # traceback.print_exc()
                    #
                    # continue

        # self.assertTrue(len(errors) == 0, '\n'.join(errors))

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
                if quantize_config.bits not in self.QLINEAR_DICT[inference_backend].SUPPORTS_BITS:
                    # Skip inference_backend that does not support the current bits
                    continue

                self.eval(inference_backend, quant_backend, quantize_config, tmp_dir)

    def eval(self, inference_backend, quant_backend, quantize_config, tmp_dir):
        print("-----------------------eval-----------------------")
        print(
            f'bits: {quantize_config.bits}, quant_backend: {quant_backend}, inference_backend: {inference_backend}. start eval')
        model = GPTQModel.load(
            tmp_dir,
            device_map="auto",
            backend=inference_backend,
        )
        results = evaluate(
            model_or_id_or_path=model,
            output_path=tmp_dir,
            tasks=[TASK_NAME],
            apply_chat_template=False,
            trust_remote_code=False,
            batch_size=4,
        )
        print('--------Eval Result---------')
        print(format_eval_result_table(results))
        print('--------Eval Result End---------')
        task_results = {
            metric: value for metric, value in get_eval_task_metrics(results, TASK_NAME).items()
            if metric != 'alias' and 'stderr' not in metric
        }
        print(f"bits is: {quantize_config.bits}, quant_backend: {quant_backend}, inference_backend: {inference_backend} -> task_results: {task_results}")
        del model

        self.check_results(quantize_config.bits, task_results)
