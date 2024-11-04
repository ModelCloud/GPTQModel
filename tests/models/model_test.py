# -- do not touch
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch
import unittest
from datasets import load_dataset  # noqa: E402
from gptqmodel import BACKEND, GPTQModel
from gptqmodel.quantization import FORMAT  # noqa: E402
from gptqmodel.quantization.config import QuantizeConfig # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
import tempfile
from lm_eval.utils import make_table

class ModelTest(unittest.TestCase):
    GENERATE_EVAL_SIZE = 100
    TASK_NAME = "gsm8k_cot"
    def generate(self, model, tokenizer, prompt=None):
        if prompt == None:
            prompt = "I am in Paris and"
        device = model.device
        inp = tokenizer(prompt, return_tensors="pt").to(device)
        res = model.generate(**inp, num_beams=1, do_sample=False, min_new_tokens=self.GENERATE_EVAL_SIZE, max_new_tokens=self.GENERATE_EVAL_SIZE)
        output = tokenizer.decode(res[0])
        print(f"Result is: \n{output}")
        return output

    def generateChat(self, model, tokenizer, prompt=None):
        if prompt == None:
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

    def load_tokenizer(self, model_name_or_path, trust_remote_code=False):
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
        return tokenizer

    def load_dataset(self, tokenizer):
        traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train").filter(lambda x: len(x['text']) >= 512)
        calibration_dataset = [tokenizer(example["text"]) for example in traindata.select(range(1024))]
        return calibration_dataset

    def quantModel(self, model_name_or_path, trust_remote_code=False, torch_dtype="auto"):
        tokenizer = self.load_tokenizer(model_name_or_path, trust_remote_code=trust_remote_code)
        calibration_dataset = self.load_dataset(tokenizer)
        quantize_config = QuantizeConfig(
            bits=4,
            group_size=128,
            format=FORMAT.GPTQ,
        )

        model = GPTQModel.from_pretrained(
            model_name_or_path,
            quantize_config=quantize_config,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype
        )

        # mpt model need
        if not model.config.pad_token_id:
            model.config.pad_token_id = tokenizer.pad_token_id or 0
        if not model.config.eos_token_id:
            model.config.eos_token_id = tokenizer.eos_token_id or 0

        model.quantize(calibration_dataset, batch_size=64)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_quantized(tmpdirname)
            q_model, q_tokenizer = self.loadQuantModel(tmpdirname, tokenizer_path=model_name_or_path)

        return q_model, q_tokenizer


    def loadQuantModel(self, model_name_or_path, trust_remote_code=False, tokenizer_path=None):
        if tokenizer_path == None:
            tokenizer_path = model_name_or_path
        else:
            trust_remote_code = True
        tokenizer = self.load_tokenizer(tokenizer_path, trust_remote_code)

        model = GPTQModel.from_quantized(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )

        return model, tokenizer

    def lm_eval(self, model, apply_chat_template=False, trust_remote_code=False):
        with tempfile.TemporaryDirectory() as tmp_dir:
            results = model.lm_eval(
                output_path=tmp_dir,
                tasks=self.TASK_NAME,
                apply_chat_template=apply_chat_template,
                trust_remote_code=trust_remote_code
            )
            task_results = {
                metric: value for metric, value in results['results'].get(self.TASK_NAME, {}).items()
                if metric != 'alias' and 'stderr' not in metric
            }
            print('--------Eval Result---------')
            print(make_table(results))
            if "groups" in results:
                print(make_table(results, "groups"))
            print('--------Eval Result End---------')
            return task_results