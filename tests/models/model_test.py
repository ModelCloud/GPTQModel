# -- do not touch
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch
import contextlib  # noqa: E402
import gc  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import unittest  # noqa: E402

import torch.cuda  # noqa: E402
from datasets import load_dataset  # noqa: E402
from gptqmodel import BACKEND, GPTQModel  # noqa: E402
from gptqmodel.quantization import FORMAT  # noqa: E402
from gptqmodel.quantization.config import QuantizeConfig  # noqa: E402
from gptqmodel.utils.eval import lm_eval  # noqa: E402
from gptqmodel.nn_modules.qlinear.marlin import MarlinQuantLinear  # noqa: E402
from gptqmodel.utils.importer import select_quant_linear  # noqa: E402
from lm_eval.utils import make_table  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

RAND_SEED = 898

class ModelTest(unittest.TestCase):
    TASK_NAME = ["arc_challenge"]
    # sub test can modify
    QUANT_ARC_MAX_DELTA_FLOOR_PERCENT = 0.15  # -15%
    QUANT_ARC_MAX_POSITIVE_DELTA_CEIL_PERCENT = 1.0  # 200%
    TRUST_REMOTE_CODE = False
    APPLY_CHAT_TEMPLATE = False
    TORCH_DTYPE = "auto"
    BATCH_SIZE = "auto"
    LOAD_BACKEND = BACKEND.AUTO
    USE_VLLM = False
    INPUTS_MAX_LENGTH = 2048
    MODEL_MAX_LEN = 4096
    DELETE_QUANTIZED_MODEL = True

    KERNEL_QUANT = {} # kernel sets
    KERNEL_INFERENCE = {} # kernel sets

    # quant config
    QUANT_FORMAT = FORMAT.GPTQ
    DESC_ACT = True
    SYM = True

    def generate(self, model, tokenizer, prompt=None):
        if prompt is None:
            prompt = "I am in Paris and"
        device = model.device
        inp = tokenizer(prompt, return_tensors="pt").to(device)
        res = model.generate(**inp, num_beams=1, do_sample=False, min_new_tokens=self.GENERATE_EVAL_SIZE,
                             max_new_tokens=self.GENERATE_EVAL_SIZE)
        output = tokenizer.decode(res[0])
        print(f"Result is: \n{output}")
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
        outputs = model.generate(input_ids=input_tensor.to(model.device), max_new_tokens=self.GENERATE_EVAL_SIZE)
        output = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
        print(f"Result is: \n{output}")
        return output

    def load_tokenizer(self, model_id_or_path, trust_remote_code=False):
        tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=trust_remote_code)
        return tokenizer

    def load_dataset(self, tokenizer):
        traindata = load_dataset("allenai/c4", data_files="en/c4-train.00001-of-01024.json.gz", split="train")
        datas = []
        for index, sample in enumerate(traindata):
            tokenized = tokenizer(sample['text'])
            if len(tokenized.data['input_ids']) < self.INPUTS_MAX_LENGTH:
                datas.append(tokenized)
                if len(datas) >= 1024:
                    break

        return datas

    def check_kernel(self, model, kernels):
        modules = set([type(module).__name__ for _, module in model.named_modules()])
        print(f"modules in model: {", ".join(modules)}")
        assert modules == kernels, f"kernels are different with expected: {", ".join(kernel)}"

    def quantModel(self, model_id_or_path, trust_remote_code=False, torch_dtype="auto", need_eval=True):
        quantize_config = QuantizeConfig(
            bits=4,
            group_size=128,
            format=self.QUANT_FORMAT,
            desc_act=self.DESC_ACT,
            sym=self.SYM,
        )
        model = GPTQModel.load(
            model_id_or_path,
            quantize_config=quantize_config,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            backend=self.LOAD_BACKEND,
            device_map={"": "cpu"} if self.LOAD_BACKEND == BACKEND.IPEX else "auto",
        )

        tokenizer = self.load_tokenizer(model_id_or_path, trust_remote_code=trust_remote_code)

        calibration_dataset = self.load_dataset(tokenizer)

        # mpt model need
        if not model.config.pad_token_id:
            model.config.pad_token_id = tokenizer.pad_token_id or 0
        if not model.config.eos_token_id:
            model.config.eos_token_id = tokenizer.eos_token_id or 0

        is_quantized = model.quantized
        if not is_quantized:
            model.quantize(calibration_dataset)

            if self.KERNEL_QUANT:
                self.check_kernel(model, self.KERNEL_QUANT)

            with (contextlib.nullcontext(tempfile.mkdtemp()) if need_eval else tempfile.TemporaryDirectory()) as tmpdirname:
                model.save(tmpdirname)
                tokenizer.save_pretrained(tmpdirname)
                q_model, q_tokenizer = self.loadQuantModel(tmpdirname, trust_remote_code=trust_remote_code)

        if not is_quantized:
            del model
            gc.collect()
            torch.cuda.empty_cache()
            return q_model, q_tokenizer
        else:
            return model, tokenizer

    def loadQuantModel(self, model_id_or_path, trust_remote_code=False, tokenizer_path=None):
        if tokenizer_path is None:
            tokenizer_path = model_id_or_path
        else:
            trust_remote_code = True
        tokenizer = self.load_tokenizer(tokenizer_path, trust_remote_code)

        model = GPTQModel.load(
            model_id_or_path,
            trust_remote_code=trust_remote_code,
            device_map={"": "cpu"} if self.LOAD_BACKEND == BACKEND.IPEX else "auto",
        )

        return model, tokenizer

    def lm_eval(self, model, apply_chat_template=False, trust_remote_code=False, delete_quantized_model=False):
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                if self.USE_VLLM:
                    model_args = f"pretrained={model.model_id_or_path},dtype=auto,gpu_memory_utilization=0.8,tensor_parallel_size=1,trust_remote_code={trust_remote_code},max_model_len={self.MODEL_MAX_LEN}"
                else:
                    model_args = ""
                results = lm_eval(
                    model,
                    model_name="vllm" if self.USE_VLLM else "hf",
                    model_args=model_args,
                    output_path=tmp_dir,
                    tasks=self.TASK_NAME,
                    apply_chat_template=apply_chat_template,
                    trust_remote_code=trust_remote_code,
                    batch_size=self.BATCH_SIZE,
                    gen_kwargs="temperature=0.0,top_k=50",
                    random_seed = RAND_SEED,
                    numpy_random_seed = RAND_SEED,
                    torch_random_seed = RAND_SEED,
                    fewshot_random_seed = RAND_SEED,
                )

                print('--------Eval Result---------')
                print(make_table(results))
                if "groups" in results:
                    print(make_table(results, "groups"))
                print('--------Eval Result End---------')
                task_results = {
                    metric: value for metric, value in results['results'].get(self.TASK_NAME, {}).items()
                    if metric != 'alias' and 'stderr' not in metric
                }
                print(task_results)
                if delete_quantized_model and os.path.exists(model.model_id_or_path):
                    shutil.rmtree(model.model_id_or_path)
                return task_results
        except BaseException as e:
            if isinstance(e, torch.OutOfMemoryError):
                old_batch = self.BATCH_SIZE
                if self.BATCH_SIZE == "auto":
                    self.BATCH_SIZE = "8"
                else:
                    self.BATCH_SIZE = f"{int(int(self.BATCH_SIZE) / 2)}"
                    self.MODEL_MAX_LEN = max(1024, self.MODEL_MAX_LEN - 1024)

                print(f"batch {old_batch} OOM, retrying with batch {self.BATCH_SIZE}")

                if int(self.BATCH_SIZE) > 0:
                    self.lm_eval(model=model,
                                 apply_chat_template=apply_chat_template,
                                 trust_remote_code=trust_remote_code,
                                 delete_quantized_model=delete_quantized_model)
                    print(f"set batch size to {self.BATCH_SIZE}, passed")
                else:
                    print(f"set batch size to {self.BATCH_SIZE}, failed")
                    raise e
            else:
                raise e

    def calculatorPer(self, filter, value):
        if "norm" in filter:
            diff_pct = (value / self.NATIVE_ARC_CHALLENGE_ACC_NORM) * 100
            print(f"{filter}: {value} diff {diff_pct:.2f}%")
        else:
            diff_pct = (value / self.NATIVE_ARC_CHALLENGE_ACC) * 100
            print(f"{filter}: {value} diff {diff_pct:.2f}%")
        return diff_pct

    def quant_lm_eval(self):
        self.model, self.tokenizer = self.quantModel(self.NATIVE_MODEL_ID, trust_remote_code=self.TRUST_REMOTE_CODE, torch_dtype=self.TORCH_DTYPE)

        if self.KERNEL_INFERENCE:
            self.check_kernel(self.model, self.KERNEL_INFERENCE)

        task_results = self.lm_eval(model=self.model,
                                    apply_chat_template=self.APPLY_CHAT_TEMPLATE,
                                    trust_remote_code=self.TRUST_REMOTE_CODE,
                                    delete_quantized_model=self.DELETE_QUANTIZED_MODEL)
        self.check_results(task_results)

    def check_results(self, task_results):
        for filter, value in task_results.items():
            diff_pct = self.calculatorPer(filter=filter, value=value)
            negative_pct = 100 * (1 - self.QUANT_ARC_MAX_DELTA_FLOOR_PERCENT)
            positive_pct = 100 * (1 + self.QUANT_ARC_MAX_POSITIVE_DELTA_CEIL_PERCENT)
            self.assertTrue(negative_pct <= diff_pct <= positive_pct, f"{filter}: {value} diff {diff_pct:.2f}% is out of the expected range [{negative_pct}-{positive_pct}%]")
