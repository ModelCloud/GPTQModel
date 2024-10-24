import unittest
from datasets import load_dataset  # noqa: E402
from gptqmodel import BACKEND, GPTQModel
from gptqmodel.quantization import FORMAT  # noqa: E402
from gptqmodel.quantization.config import QuantizeConfig # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
import tempfile

class BaseTest(unittest.TestCase):
    GENERATE_EVAL_SIZE = 100

    def generate(self, model, tokenizer):
        prompt = "I am in Paris and"
        device = model.device
        inp = tokenizer(prompt, return_tensors="pt").to(device)
        res = model.generate(**inp, num_beams=1, do_sample=False, min_new_tokens=self.GENERATE_EVAL_SIZE, max_new_tokens=self.GENERATE_EVAL_SIZE)
        output = tokenizer.decode(res[0])

        return output

    def load_tokenizer(self, model_name_or_path, trust_remote_code=False):
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
        return tokenizer

    def load_dataset(self, tokenizer):
        traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train").filter(lambda x: len(x['text']) >= 512)
        calibration_dataset = [tokenizer(example["text"]) for example in traindata.select(range(1024))]
        return calibration_dataset

    def quantModel(self, model_name_or_path, trust_remote_code=False):
        tokenizer = self.load_tokenizer(model_name_or_path)
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
        )

        model.quantize(calibration_dataset, batch_size=64)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model.save_quantized(tmpdirname)
            q_model, q_tokenizer = self.loadQuantModel(tmpdirname)

        return q_model, q_tokenizer


    def loadQuantModel(self, model_name_or_path, trust_remote_code=False):
        tokenizer = self.load_tokenizer(model_name_or_path, trust_remote_code)

        model = GPTQModel.from_quantized(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )

        return model, tokenizer