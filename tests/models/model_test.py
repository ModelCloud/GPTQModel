# -- do not touch
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch
import shutil  # noqa: E402
import tempfile  # noqa: E402
import unittest  # noqa: E402

from datasets import load_dataset  # noqa: E402
from gptqmodel import GPTQModel  # noqa: E402
from gptqmodel.quantization import FORMAT  # noqa: E402
from gptqmodel.quantization.config import QuantizeConfig  # noqa: E402
from lm_eval.utils import make_table  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


class ModelTest(unittest.TestCase):
    TASK_NAME = "arc_challenge"
    # sub test can modify
    QUANT_ARC_MAX_NEGATIVE_DELTA = 0.1  # -10%
    QUANT_ARC_MAX_POSITIVE_DELTA = 0.2  # 20%
    TRUST_REMOTE_CODE = False
    APPLY_CHAT_TEMPLATE = False
    TORCH_DTYPE = "auto"

    def generate(self, model, tokenizer, prompt=None):
        if prompt is None:
            prompt = "I am in Paris and"
        device = model.device
        inp = tokenizer(prompt, return_tensors="pt").to(device)
        res = model.generate(**inp, num_beams=1, do_sample=False, min_new_tokens=self.GENERATE_EVAL_SIZE, max_new_tokens=self.GENERATE_EVAL_SIZE)
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
        max_length = 2048
        traindata = load_dataset("allenai/c4", data_files="en/c4-train.00001-of-01024.json.gz",
                                 split="train").filter(
            lambda x: len(x["text"]) >= max_length and len(x["text"]) <= (max_length * 1.5))
        return [tokenizer(example["text"]) for example in traindata.select(range(1024))]

    def quantModel(self, model_id_or_path, trust_remote_code=False, torch_dtype="auto", need_eval=True):
        tokenizer = self.load_tokenizer(model_id_or_path, trust_remote_code=trust_remote_code)
        calibration_dataset = self.load_dataset(tokenizer)
        quantize_config = QuantizeConfig(
            bits=4,
            group_size=128,
            format=FORMAT.GPTQ,
        )

        model = GPTQModel.load(
            model_id_or_path,
            quantize_config=quantize_config,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype
        )

        # mpt model need
        if not model.config.pad_token_id:
            model.config.pad_token_id = tokenizer.pad_token_id or 0
        if not model.config.eos_token_id:
            model.config.eos_token_id = tokenizer.eos_token_id or 0

        model.quantize(calibration_dataset, batch_size=4)
        if need_eval:
            test_dir = os.path.dirname(os.path.abspath(__file__))
            save_dir = os.path.join(test_dir, "test_quantized_model")
            os.makedirs(save_dir, exist_ok=True)
            model.save(save_dir)
            tokenizer.save_pretrained(save_dir)
            q_model, q_tokenizer = self.loadQuantModel(save_dir, trust_remote_code=trust_remote_code)
        else:
            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save(tmpdirname)
                tokenizer.save_pretrained(tmpdirname)
                q_model, q_tokenizer = self.loadQuantModel(tmpdirname, trust_remote_code=trust_remote_code)
        del model
        return q_model, q_tokenizer


    def loadQuantModel(self, model_id_or_path, trust_remote_code=False, tokenizer_path=None):
        if tokenizer_path is None:
            tokenizer_path = model_id_or_path
        else:
            trust_remote_code = True
        tokenizer = self.load_tokenizer(tokenizer_path, trust_remote_code)

        model = GPTQModel.load(
            model_id_or_path,
            trust_remote_code=trust_remote_code,
        )

        return model, tokenizer

    def lm_eval(self, model, apply_chat_template=False, trust_remote_code=False):
        with tempfile.TemporaryDirectory() as tmp_dir:
            results = model.lm_eval(
                model="vllm",
                model_args=f"pretrained={model.model_id_or_path},dtype=auto,gpu_memory_utilization=0.8,trust_remote_code={trust_remote_code}",
                output_path=tmp_dir,
                tasks=self.TASK_NAME,
                apply_chat_template=apply_chat_template,
                trust_remote_code=trust_remote_code,
                batch_size=8,
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
            if os.path.exists(model.model_id_or_path):
                shutil.rmtree(model.model_id_or_path)
            return task_results

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

        task_results = self.lm_eval(self.model, trust_remote_code=self.TRUST_REMOTE_CODE, apply_chat_template=self.APPLY_CHAT_TEMPLATE)
        for filter, value in task_results.items():
            diff_pct = self.calculatorPer(filter=filter, value=value)
            negative_pct = 100 * (1 - self.QUANT_ARC_MAX_NEGATIVE_DELTA)
            positive_pct = 100 * (1 + self.QUANT_ARC_MAX_POSITIVE_DELTA)
            self.assertTrue(negative_pct <= diff_pct <= positive_pct,
                            f"{filter}: {value} diff {diff_pct:.2f}% is out of the expected range [{negative_pct}-{positive_pct}%]")
