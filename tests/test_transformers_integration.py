import tempfile

from gptqmodel.integration import integration
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

from models.model_test import ModelTest


class TestTransformersIntegration(ModelTest):

    @classmethod
    def setUpClass(self):
        integration.patch_hf()

    def _test_load_quantized_model_gptq_v1(self, device_map):
        model_id_or_path = "/monster/data/model/TinyLlama-1.1B-Chat-v1.0"
        tokenizer = AutoTokenizer.from_pretrained(model_id_or_path)
        model = AutoModelForCausalLM.from_pretrained(model_id_or_path, device_map=device_map)

        generate_str = tokenizer.decode(model.generate(**tokenizer("The capital of France is is", return_tensors="pt").to(model.device), max_new_tokens=1, temperature=0, top_p=0.95, top_k=50)[0])

        self.assertIn("paris" ,generate_str.lower())

    def _test_load_quantized_model_gptq_v2(self, device_map):
        model_id_or_path = "/monster/data/model/TinyLlama-1.1B-Chat-v1.0"
        model = AutoModelForCausalLM.from_pretrained(model_id_or_path, device_map=device_map)

        tokenizer = AutoTokenizer.from_pretrained(model_id_or_path)

        generate_str = tokenizer.decode(model.generate(**tokenizer("The capital of France is is", return_tensors="pt").to(model.device), max_new_tokens=1, temperature=0, top_p=0.95, top_k=50)[0])
        self.assertIn("paris" ,generate_str.lower())

    def _test_quantize(self, device_map):
        model_id = "/monster/data/model/opt-125m"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        dataset = ["gptqmodel is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."]
        gptq_config = GPTQConfig(bits=4, dataset=dataset, tokenizer=tokenizer)
        quantized_model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device_map, quantization_config=gptq_config)

        with tempfile.TemporaryDirectory() as tmp_dir:
            quantized_model.save_pretrained(tmp_dir)
            tokenizer.save_pretrained(tmp_dir)
            del quantized_model

            model = AutoModelForCausalLM.from_pretrained(tmp_dir, device_map=device_map)

            generate_str = tokenizer.decode(model.generate(**tokenizer("gptqmodel is", return_tensors="pt").to(model.device))[0])

            self.assertIn("is a good way" ,generate_str.lower())

    def test_load_quantized_model_gptq_v1_ipex(self):
        self._test_load_quantized_model_gptq_v1(device_map="cpu")

    def test_load_quantized_model_gptq_v1_cuda(self):
        self._test_load_quantized_model_gptq_v1(device_map="cuda")

    def test_load_quantized_model_gptq_v2_ipex(self):
        self._test_load_quantized_model_gptq_v2(device_map="cpu")

    def test_load_quantized_model_gptq_v2_cuda(self):
        self._test_load_quantized_model_gptq_v2(device_map="cuda")

    def test_quantize_ipex(self):
        self._test_quantize(device_map="cpu")

    def test_quantize_cuda(self):
        self._test_quantize(device_map="cuda")
