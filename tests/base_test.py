import unittest
from datasets import load_dataset  # noqa: E402
from gptqmodel import BACKEND, GPTQModel
from gptqmodel.quantization import FORMAT  # noqa: E402
from gptqmodel.quantization.config import QuantizeConfig # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

class BaseTest(unittest.TestCase):
    GENERATE_EVAL_SIZE = 100

    def generate(self, model, tokenizer):
        prompt = "I am in Paris and"

        inp = tokenizer(prompt, return_tensors="pt").to(model.device)
        res = model.generate(**inp, num_beams=1, do_sample=False, min_new_tokens=60, max_new_tokens=60)
        output = tokenizer.decode(res[0])

        return output

    def loadModel(self, model_name_or_path, quant=False):
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train").filter(lambda x: len(x['text']) >= 512)
        calibration_dataset = [tokenizer(example["text"]) for example in traindata.select(range(1024))]

        quantize_config = QuantizeConfig(
            bits=4,
            group_size=128,
            format=FORMAT.GPTQ_V2,
        )
        model = GPTQModel.from_pretrained(
            model_name_or_path,
            quantize_config=quantize_config,
            trust_remote_code=True,
        )

        if quant:
            quantized_model = model.quantize(calibration_dataset, batch_size=64)
            return quantized_model, tokenizer
        else:
            return model, tokenizer

    def loadQuantModel(self, model_name_or_path):
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

        model = GPTQModel.from_quantized(
            model_name_or_path,
            trust_remote_code=True,
        )

        return model, tokenizer