# -- do not touch
import importlib
import logging
import os
import pkgutil
import traceback

from gptqmodel.nn_modules.qlinear import BaseQuantLinear
from transformers import AutoTokenizer

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch
import tempfile  # noqa: E402
import unittest  # noqa: E402

from gptqmodel import GPTQModel, QuantizeConfig, BACKEND  # noqa: E402
from gptqmodel.utils.eval import lm_eval  # noqa: E402

from gptqmodel.nn_modules.qlinear.bitblas import BitBLASQuantLinear
from gptqmodel.nn_modules.qlinear.dynamic_cuda import DynamicCudaQuantLinear
from gptqmodel.nn_modules.qlinear.exllama import ExllamaQuantLinear
from gptqmodel.nn_modules.qlinear.exllamav2 import ExllamaV2QuantLinear
from gptqmodel.nn_modules.qlinear.ipex import IPEXQuantLinear
from gptqmodel.nn_modules.qlinear.marlin import MarlinQuantLinear
from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear
from gptqmodel.nn_modules.qlinear.tritonv2 import TritonV2QuantLinear

from lm_eval.utils import make_table  # noqa: E402

logger = logging.getLogger(__name__)

RAND_SEED = 42
TASK_NAME = "arc_challenge"


def get_all_qlinear():
    qlinear_classes = []
    package_name = "gptqmodel.nn_modules.qlinear"
    package = importlib.import_module(package_name)
    for _, name, ispkg in pkgutil.iter_modules(package.__path__, package_name + "."):
        if not ispkg:
            try:
                module = importlib.import_module(name)

                for class_name, class_obj in module.__dict__.items():
                    if isinstance(class_obj, type):
                        if issubclass(class_obj, BaseQuantLinear) and class_obj is not BaseQuantLinear:
                            logger.info(f"Found class {class_name} in {name} that inherits from BaseClass")
                            qlinear_classes.append(class_obj)
            except Exception:
                continue
    return qlinear_classes


class TestGroupSize(unittest.TestCase):
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
    @classmethod
    def setUpClass(cls):
        cls.pack_backends = [BACKEND.EXLLAMA_V1, BACKEND.TRITON, BACKEND.CUDA, BACKEND.TORCH, BACKEND.BITBLAS,
                             BACKEND.IPEX]
        cls.backends = list(cls.pack_backends)
        cls.backends.extend([BACKEND.EXLLAMA_V2, BACKEND.MARLIN, ])

    def test_group_size(self):
        # quantize
        group_sizes = [-1, 16, 32, 64, 128]
        TINYLLAMA_MODEL_ID = "/monster/data/model/tinyllama-15M-stories"
        OPT_MODEL_ID = "/monster/data/model/opt-125m"
        model_id = OPT_MODEL_ID
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        dataset = [
            "gptqmodel is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."]
        calibration_dataset = [tokenizer(example) for example in dataset]
        for group_size in group_sizes:
            quantize_config = QuantizeConfig(bits=4, group_size=group_size, sym=True, desc_act=False)
            for quant_backend in self.pack_backends:
                print("-----------------------quant-----------------------")
                print(f"{quantize_config.group_size}, quant_backend: {quant_backend} start quant")
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
            f"group_size is: {quantize_config.group_size}, quant_backend: {quant_backend}, inference_backend: {inference_backend} -> task_results: {task_results}")
        del model
