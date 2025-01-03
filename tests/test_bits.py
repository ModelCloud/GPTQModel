# -- do not touch
import os


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch
import logging  # noqa: E402
import tempfile  # noqa: E402
import traceback  # noqa: E402
import unittest  # noqa: E402

from lm_eval.utils import make_table  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

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


    @classmethod
    def setUpClass(cls):
        # cls.pack_backends = [BACKEND.EXLLAMA_V1, BACKEND.TRITON, BACKEND.CUDA, BACKEND.TORCH, BACKEND.BITBLAS,
        #                      BACKEND.IPEX]
        # cls.backends = list(cls.pack_backends)
        # cls.backends.extend([BACKEND.EXLLAMA_V2, BACKEND.MARLIN, ])

        # TODO Only CUDA Quant Linear is tested for now
        cls.pack_backends = [BACKEND.CUDA]
        cls.backends = list(cls.pack_backends)

    def test_group_size(self):
        # quantize
        OPT_MODEL_ID = "/monster/data/model/opt-125m"
        model_id = OPT_MODEL_ID
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
