# Copyright 2024-2025 ModelCloud.ai
# Copyright 2024-2025 qubitium@modelcloud.ai
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
import sys

if sys.platform == "darwin":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# -- end do not touch

from pathlib import Path  # noqa: E402
from typing import Dict, List  # noqa: E402

from logbar import LogBar  # noqa: E402

sys.path.insert(0, f"{str(Path(__file__).resolve().parent.parent)}/models")  # noqa: E402
import contextlib  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import unittest  # noqa: E402

import torch.cuda  # noqa: E402
import transformers  # noqa: E402
from datasets import load_dataset  # noqa: E402
from gptqmodel import BACKEND, GPTQModel  # noqa: E402
from gptqmodel.nn_modules.qlinear import BaseQuantLinear  # noqa: E402
from gptqmodel.quantization import FORMAT  # noqa: E402
from gptqmodel.quantization.config import QuantizeConfig  # noqa: E402
from gptqmodel.utils.eval import EVAL  # noqa: E402
from gptqmodel.utils.model import MODALITY  # noqa: E402
from gptqmodel.utils.torch import torch_empty_cache  # noqa: E402
from ovis.image_to_test_dataset import get_calib_dataset  # noqa: E402
from packaging.version import Version  # noqa: E402
from transformers import AutoProcessor, AutoTokenizer  # noqa: E402

RAND_SEED = 898

log = LogBar.shared()

class ModelTest(unittest.TestCase):
    TASK_NAME = EVAL.LM_EVAL.ARC_CHALLENGE
    # sub test can modify
    QUANT_ARC_MAX_DELTA_FLOOR_PERCENT = 0.15  # -15%
    QUANT_ARC_MAX_POSITIVE_DELTA_CEIL_PERCENT = 1.0  # 200%
    TRUST_REMOTE_CODE = False
    APPLY_CHAT_TEMPLATE = False
    TORCH_DTYPE = "auto"
    EVAL_BATCH_SIZE = "auto"
    QUANT_BATCH_SIZE = 1
    LOAD_BACKEND = BACKEND.AUTO
    QUANT_BACKEND = BACKEND.AUTO
    USE_VLLM = False
    INPUTS_MAX_LENGTH = 2048
    MODEL_MAX_LEN = 4096
    DATASET_SIZE = 256
    DELETE_QUANTIZED_MODEL = True

    KERNEL_QUANT = {}  # kernel sets
    KERNEL_INFERENCE = {}  # kernel sets

    # quant config
    QUANT_FORMAT = FORMAT.GPTQ
    DESC_ACT = True
    SYM = True

    DISABLE_FLASH_ATTN = False
    LOAD_QUANTIZED_MODEL = None  # loading from a quantized dir instead of using native model id/dir
    SAVE_QUANTIZED_MODEL = None  # if quantize a model, save it to this dir

    INFERENCE_PROMPT = "The capital city of France is named"
    INFERENCE_RESULT_KEYWORDS = ["paris"]
    GENERATE_EVAL_SIZE_MIN = 128
    GENERATE_EVAL_SIZE_MAX = 128

    LM_HEAD_LOSS_MAX_DELTA_PERCENT = 0.1  # ±10%
    EXPECT_LM_HEAD_LOSS = None

    QUANTIZE_CONFIG_BITS = 4
    QUANTIZE_CONFIG_GROUP_SIZE = 128

    V2 = False

    def assertInference(self, model, tokenizer=None, keywords=None, prompt=INFERENCE_PROMPT):
        # gptqmodel can auto init tokenizer internally
        if keywords is None:
            keywords = self.INFERENCE_RESULT_KEYWORDS
        if tokenizer is None:
            tokenizer = model.tokenizer

        generated = self.generate(model, tokenizer, prompt).lower()
        for k in keywords:
            if k.lower() in generated:
                self.assertTrue(True)
                return
        self.assertTrue(False, f"none of keywords were found in generated: `{generated}`")

    # note that sampling is disabled for help with deterministic generation for ci tests
    def generate(self, model, tokenizer, prompt=None):
        if prompt is None:
            prompt = self.INFERENCE_PROMPT
        inp = tokenizer(prompt, return_tensors="pt").to(model.device)
        res = model.generate(**inp, num_beams=1, do_sample=False, min_new_tokens=self.GENERATE_EVAL_SIZE_MIN, max_new_tokens=self.GENERATE_EVAL_SIZE_MIN)
        output = tokenizer.decode(res[0])
        print(f"Result is: >>\n{output}\n<<")
        return output

    def generateChat(self, model, tokenizer, prompt=None):
        if prompt is None:
            prompt = [
                {"role": "system",
                 "content": "You are a helpful assistant."},
                {"role": "user",
                 "content": "I am in Shanghai, preparing to visit the natural history museum. Can you tell me the best way to"}
            ]

        input_tensor = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, return_tensors="pt")
        outputs = model.generate(input_ids=input_tensor.to(model.device), max_new_tokens=self.GENERATE_EVAL_SIZE_MAX)
        output = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
        print(f"Result is: \n{output}")
        return output

    def load_tokenizer(self, model_id_or_path, trust_remote_code=False):
        tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=trust_remote_code)
        return tokenizer

    @classmethod
    def load_dataset(self, tokenizer=None, rows: int = DATASET_SIZE):
        traindata = load_dataset("json", data_files="/monster/data/model/dataset/c4-train.00000-of-01024.json.gz", split="train")

        if not tokenizer:
            return traindata.select(range(rows))

        datas = []
        for index, sample in enumerate(traindata):
            tokenized = tokenizer(sample['text'])
            if len(tokenized.data['input_ids']) < self.INPUTS_MAX_LENGTH:
                datas.append(tokenized)
                if len(datas) >= rows:
                    break

        return datas

    def check_kernel(self, model, expected_kernels):
        modules = {module.__class__ for _, module in model.named_modules() if isinstance(module, BaseQuantLinear)}
        print(f"modules in model: {modules}")
        if expected_kernels:
            assert modules == expected_kernels, f"kernels are different with expected. found: {modules}. expected: {expected_kernels}"

    def quantModel(self, model_id_or_path, trust_remote_code=False, torch_dtype="auto", need_eval=True, batch_size: int = QUANT_BATCH_SIZE, **kwargs):
        quantize_config = QuantizeConfig(
            bits=self.QUANTIZE_CONFIG_BITS,
            group_size=self.QUANTIZE_CONFIG_GROUP_SIZE,
            format=self.QUANT_FORMAT,
            desc_act=self.DESC_ACT,
            sym=self.SYM,
            v2=self.V2
        )

        log.info(f"Quant config: {quantize_config}")
        log.info(f"Quant batch_size: {batch_size}")

        args = kwargs if kwargs else {}

        if self.DISABLE_FLASH_ATTN:
            has_attn_implementation = Version(transformers.__version__) >= Version("4.46.0")
            if has_attn_implementation:
                args["attn_implementation"] = "eager"
            args["use_flash_attention_2"] = False

        log.info(f"args: {args}")
        model = GPTQModel.load(
            model_id_or_path,
            quantize_config=quantize_config,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            device_map={"": "cpu"} if self.LOAD_BACKEND == BACKEND.IPEX else "auto",
            **args,
        )

        tokenizer = self.load_tokenizer(model_id_or_path, trust_remote_code=trust_remote_code)

        is_image_to_text_model = MODALITY.IMAGE_TO_TEXT in model.modality
        calibration_dataset = get_calib_dataset(model) if is_image_to_text_model else self.load_dataset(tokenizer)

        # mpt model need
        if not model.config.pad_token_id:
            model.config.pad_token_id = tokenizer.pad_token_id or 0
        if not model.config.eos_token_id:
            model.config.eos_token_id = tokenizer.eos_token_id or 0

        is_quantized = model.quantized

        # ovis cannot load processor
        is_ovis_model = model.__class__.__name__ == "OvisGPTQ"
        need_create_processor = is_image_to_text_model and not is_ovis_model
        if not is_quantized:
            model.quantize(calibration_dataset, backend=self.QUANT_BACKEND, batch_size=batch_size)

            self.check_kernel(model, self.KERNEL_QUANT)

            with (contextlib.nullcontext(self.SAVE_QUANTIZED_MODEL) if self.SAVE_QUANTIZED_MODEL else contextlib.nullcontext(tempfile.mkdtemp()) if need_eval else tempfile.TemporaryDirectory()) as tmpdirname:
                os.makedirs(tmpdirname, exist_ok=True)
                self.clear_directory(tmpdirname)

                model.save(tmpdirname)
                tokenizer.save_pretrained(tmpdirname)
                q_model, q_tokenizer = self.loadQuantModel(tmpdirname, trust_remote_code=trust_remote_code)
                if need_create_processor:
                    processor = AutoProcessor.from_pretrained(tmpdirname)
        else:
            if need_create_processor:
                processor = AutoProcessor.from_pretrained(model_id_or_path)
        if not is_quantized:
            del model
            torch_empty_cache()
            if need_create_processor:
                return q_model, q_tokenizer, processor
            else:
                return q_model, q_tokenizer
        else:
            if need_create_processor:
                return model, tokenizer, processor
            else:
                return model, tokenizer

    def loadQuantModel(self, model_id_or_path, trust_remote_code=False, tokenizer_path=None, **args):
        if tokenizer_path is None:
            tokenizer_path = model_id_or_path
        else:
            trust_remote_code = True
        tokenizer = self.load_tokenizer(tokenizer_path, trust_remote_code)

        kargs = args if args else {}

        if self.DISABLE_FLASH_ATTN:
            has_attn_implementation = Version(transformers.__version__) >= Version("4.46.0")
            if has_attn_implementation:
                kargs["attn_implementation"] = None
            kargs["use_flash_attention_2"] = False

        model = GPTQModel.load(
            model_id_or_path,
            trust_remote_code=trust_remote_code,
            backend=self.LOAD_BACKEND,
            device_map={"": "cpu"} if self.LOAD_BACKEND == BACKEND.IPEX else "auto",
            **kargs
        )

        return model, tokenizer

    def lm_eval(self, model, apply_chat_template=False, trust_remote_code=False, delete_quantized_model=False, extra_args:dict=None):
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                if self.USE_VLLM:
                    model_args = {
                        "pretrained": model.model_local_path,
                        "dtype": "auto", #"float16",
                        "gpu_memory_utilization": 0.8,
                        "tensor_parallel_size": 1,
                        "trust_remote_code": trust_remote_code,
                        "max_model_len": self.MODEL_MAX_LEN
                    }
                else:
                    model_args = {}
                if extra_args:
                    model_args.update(extra_args)

                from lm_eval.tasks import TaskManager
                from lm_eval.utils import make_table

                task_groups = EVAL.get_task_groups_from_tasks(self.TASK_NAME)

                for framework, tasks in task_groups.items():
                    log.info(f"TEST: EVAL starting: backend = {self.LOAD_BACKEND}")
                    results = GPTQModel.eval(
                        model_or_id_or_path=model,
                        llm_backend="vllm" if self.USE_VLLM else "gptqmodel",
                        model_args=model_args,
                        output_path=tmp_dir,
                        framework=framework,
                        tasks=tasks,
                        apply_chat_template=apply_chat_template,
                        trust_remote_code=trust_remote_code,
                        batch_size=self.EVAL_BATCH_SIZE,
                        gen_kwargs="temperature=0.0,top_k=50",
                        random_seed=RAND_SEED,
                        task_manager=TaskManager(include_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../tasks"), include_defaults=False)
                    )

                    print('--------Eval Result---------')
                    print(make_table(results))
                    if "groups" in results:
                        print(make_table(results, "groups"))
                    print('--------Eval Result End---------')
                    task_results = {
                        metric: value for metric, value in results['results'].get(self.TASK_NAME.value, {}).items()
                        if metric != 'alias' and 'stderr' not in metric
                    }
                    print(task_results)

                if delete_quantized_model and model.model_local_path.startswith("/tmp") and os.path.exists(
                        model.model_local_path):
                    shutil.rmtree(model.model_local_path)
                return task_results
        except BaseException as e:
            if isinstance(e, torch.OutOfMemoryError):
                old_batch = self.EVAL_BATCH_SIZE
                if self.EVAL_BATCH_SIZE == "auto":
                    self.EVAL_BATCH_SIZE = "8"
                else:
                    self.EVAL_BATCH_SIZE = f"{int(int(self.EVAL_BATCH_SIZE) / 2)}"
                    self.MODEL_MAX_LEN = max(1024, self.MODEL_MAX_LEN - 1024)

                print(f"batch {old_batch} OOM, retrying with batch {self.EVAL_BATCH_SIZE}")

                if int(self.EVAL_BATCH_SIZE) > 0:
                    self.lm_eval(model=model,
                                 apply_chat_template=apply_chat_template,
                                 trust_remote_code=trust_remote_code,
                                 delete_quantized_model=delete_quantized_model)
                    print(f"set batch size to {self.EVAL_BATCH_SIZE}, passed")
                else:
                    print(f"set batch size to {self.EVAL_BATCH_SIZE}, failed")
                    raise e
            else:
                raise e

    def calculatorPer(self, filter, value):
        if "norm" in filter:
            expected = self.NATIVE_ARC_CHALLENGE_ACC_NORM
        else:
            expected = self.NATIVE_ARC_CHALLENGE_ACC

        diff_pct = (value / expected) * 100
        log.info(f"{filter}: `{value}` vs `{expected}` diff {diff_pct:.2f}%")

        return diff_pct, expected

    def quant_lm_eval(self):
        self.model = None
        if self.LOAD_QUANTIZED_MODEL:
            try:
                self.model, _ = self.quantModel(self.SAVE_QUANTIZED_MODEL, batch_size=self.QUANT_BATCH_SIZE, trust_remote_code=self.TRUST_REMOTE_CODE, torch_dtype=self.TORCH_DTYPE)
            except BaseException as e:
                print(f"LOAD_QUANTIZED_MODEL: {self.LOAD_QUANTIZED_MODEL} has something wrong {e}\n use NATIVE_MODEL_ID: {self.NATIVE_MODEL_ID} instead")
        if not self.model:
            self.model, _ = self.quantModel(self.NATIVE_MODEL_ID, batch_size=self.QUANT_BATCH_SIZE, trust_remote_code=self.TRUST_REMOTE_CODE, torch_dtype=self.TORCH_DTYPE)

        self.check_kernel(self.model, self.KERNEL_INFERENCE)

        task_results = self.lm_eval(model=self.model,
                                    apply_chat_template=self.APPLY_CHAT_TEMPLATE,
                                    trust_remote_code=self.TRUST_REMOTE_CODE,
                                    delete_quantized_model=self.DELETE_QUANTIZED_MODEL)
        self.check_results(task_results)

    def check_results(self, task_results):
        for filter, value in task_results.items():
            diff_pct, expected = self.calculatorPer(filter=filter, value=value)
            negative_pct = 100 * (1 - self.QUANT_ARC_MAX_DELTA_FLOOR_PERCENT)
            positive_pct = 100 * (1 + self.QUANT_ARC_MAX_POSITIVE_DELTA_CEIL_PERCENT)
            self.assertTrue(negative_pct <= diff_pct <= positive_pct, f"{filter}: `{value}` vs expected `{expected}`, diff {diff_pct:.2f}% is out of the expected range [{negative_pct}-{positive_pct}%]")

    def check_lm_head_loss(self, quant_log: List[Dict[str, any]]):
        final_log = quant_log[-1]
        if final_log["module"] == "lm_head":
            loss_value = float(final_log["loss"])
            diff_pct = (loss_value / self.EXPECT_LM_HEAD_LOSS) * 100
            print(f"lm_head loss: {loss_value} diff {diff_pct:.2f}%")
            negative_pct = 100 * (1 - self.LM_HEAD_LOSS_MAX_DELTA_PERCENT)
            positive_pct = 100 * (1 + self.LM_HEAD_LOSS_MAX_DELTA_PERCENT)
            self.assertTrue(negative_pct <= diff_pct <= positive_pct,
                            f"lm_head loss: {loss_value} diff {diff_pct:.2f}% is out of the expected range [{negative_pct}-{positive_pct}%]")
        else:
            raise ValueError("No quantization for lm_head module")

    def clear_directory(self, directory_path):
        for item in os.listdir(directory_path):
            item_path = os.path.join(directory_path, item)
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
