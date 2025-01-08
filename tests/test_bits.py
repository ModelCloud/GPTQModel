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
import logging  # noqa: E402
import tempfile  # noqa: E402
import traceback  # noqa: E402
import unittest  # noqa: E402

from gptqmodel import BACKEND, GPTQModel, QuantizeConfig  # noqa: E402
from gptqmodel.nn_modules.qlinear.bitblas import BitBLASQuantLinear  # noqa: E402
from gptqmodel.nn_modules.qlinear.dynamic_cuda import DynamicCudaQuantLinear  # noqa: E402
from gptqmodel.nn_modules.qlinear.exllama import ExllamaQuantLinear  # noqa: E402
from gptqmodel.nn_modules.qlinear.exllamav2 import ExllamaV2QuantLinear  # noqa: E402
from gptqmodel.nn_modules.qlinear.ipex import IPEXQuantLinear  # noqa: E402
from gptqmodel.nn_modules.qlinear.marlin import MarlinQuantLinear  # noqa: E402
from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear  # noqa: E402
from gptqmodel.nn_modules.qlinear.tritonv2 import TritonV2QuantLinear  # noqa: E402
from gptqmodel.utils.eval import lm_eval  # noqa: E402
from lm_eval.utils import make_table  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

logger = logging.getLogger(__name__)

RAND_SEED = 42
TASK_NAME = "arc_challenge"

class TestBits(unittest.TestCase):
    QLINEAR_DICT = {
        BACKEND.EXLLAMA_V1: ExllamaQuantLinear,
        BACKEND.EXLLAMA_V2: ExllamaV2QuantLinear,
        BACKEND.TRITON: TritonV2QuantLinear,
        BACKEND.CUDA: DynamicCudaQuantLinear,
        BACKEND.TORCH: TorchQuantLinear,
        BACKEND.BITBLAS: BitBLASQuantLinear,
        BACKEND.IPEX: IPEXQuantLinear,
        BACKEND.MARLIN: MarlinQuantLinear,
    }

    QUANT_ARC_MAX_DELTA_FLOOR_PERCENT = 0.025  # -2.5%
    QUANT_ARC_MAX_POSITIVE_DELTA_CEIL_PERCENT = 0.025  # +2.5%

    CUDA_QLINEAR_QUANTIZED_MODEL_ARC_CHALLENGE_EXPECTS = {
        2: {'acc,none': 0.22610921501706485, 'acc_norm,none': 0.2909556313993174},
        3: {'acc,none': 0.21245733788395904, 'acc_norm,none': 0.24744027303754265},
        4: {'acc,none': 0.2738907849829352, 'acc_norm,none': 0.3122866894197952},
        8: {'acc,none': 0.2841296928327645, 'acc_norm,none': 0.302901023890785},
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
            self.assertTrue(negative_pct <= diff_pct <= positive_pct, f"{filter}: {value} diff {diff_pct:.2f}% is out of the expected range [{negative_pct}-{positive_pct}%]")

    @classmethod
    def setUpClass(cls):
        # cls.pack_backends = [BACKEND.EXLLAMA_V1, BACKEND.TRITON, BACKEND.CUDA, BACKEND.TORCH, BACKEND.BITBLAS,
        #                      BACKEND.IPEX]
        # cls.backends = list(cls.pack_backends)
        # cls.backends.extend([BACKEND.EXLLAMA_V2, BACKEND.MARLIN, ])

        # TODO Only CUDA Quant Linear is tested for now
        cls.pack_backends = [BACKEND.CUDA]
        cls.backends = list(cls.pack_backends)

    def test_bits(self):
        # quantize
        model_id = "/monster/data/model/TinyLlama-1.1B-Chat-v1.0"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        dataset = [
            "gptqmodel is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."]
        calibration_dataset = [tokenizer(example) for example in dataset]
        for quant_backend in self.pack_backends:
            supports_bits = self.QLINEAR_DICT[quant_backend].SUPPORTS_BITS
            for bits in supports_bits:
                print("-----------------------quant-----------------------")
                quantize_config = QuantizeConfig(bits=bits, group_size=128, sym=True, desc_act=False)
                print(f"bits: {quantize_config.bits}, quant_backend: {quant_backend} start quant")
                try:
                    self.quant_and_eval(calibration_dataset, model_id, quant_backend, quantize_config, tokenizer)
                except Exception:
                    print(f"bits:  {quantize_config.bits}, quant_backend: {quant_backend} An error occurred")
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
                if quantize_config.bits not in self.QLINEAR_DICT[inference_backend].SUPPORTS_BITS:
                    # Skip inference_backend that does not support the current bits
                    continue

                try:
                    self.eval(inference_backend, quant_backend, quantize_config, tmp_dir)
                except Exception:
                    traceback.print_exc()
                    continue

    def eval(self, inference_backend, quant_backend, quantize_config, tmp_dir):
        print("-----------------------eval-----------------------")
        print(
            f'bits: {quantize_config.bits}, quant_backend: {quant_backend}, inference_backend: {inference_backend}. start eval')
        model = GPTQModel.load(
            tmp_dir,
            device_map="auto",
            backend=inference_backend,
        )
        results = lm_eval(
            model,
            model_name="hf",
            output_path=tmp_dir,
            tasks=TASK_NAME,
            apply_chat_template=False,
            trust_remote_code=False,
            batch_size=32,
            gen_kwargs="temperature=0.0,top_k=50",
            random_seed=RAND_SEED,
            numpy_random_seed=RAND_SEED,
            torch_random_seed=RAND_SEED,
            fewshot_random_seed=RAND_SEED,
        )
        print('--------Eval Result---------')
        print(make_table(results))
        if "groups" in results:
            print(make_table(results, "groups"))
        print('--------Eval Result End---------')
        task_results = {
            metric: value for metric, value in results['results'].get(TASK_NAME, {}).items()
            if metric != 'alias' and 'stderr' not in metric
        }
        print(
            f"bits is: {quantize_config.bits}, quant_backend: {quant_backend}, inference_backend: {inference_backend} -> task_results: {task_results}")
        del model

        self.check_results(quantize_config.bits, task_results)
