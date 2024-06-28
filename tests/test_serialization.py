# -- do not touch
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

import json  # noqa: E402
import os  # noqa: E402
import tempfile  # noqa: E402
import unittest  # noqa: E402

from gptqmodel import Backend, GPTQModel  # noqa: E402
from gptqmodel.quantization import FORMAT, FORMAT_FIELD_JSON, QUANT_CONFIG_FILENAME  # noqa: E402


class TestSerialization(unittest.TestCase):
    MODEL_ID = "LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"

    def test_marlin_local_serialization(self):
        model = GPTQModel.from_quantized(self.MODEL_ID, device="cuda:0", backend=Backend.MARLIN)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_pretrained(tmpdir)

            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "gptq_model-4bit-128g.safetensors")))

            with open(os.path.join(tmpdir, QUANT_CONFIG_FILENAME), "r") as config_file:
                config = json.load(config_file)

            self.assertTrue(config[FORMAT_FIELD_JSON] == FORMAT.MARLIN)

            model = GPTQModel.from_quantized(tmpdir, device="cuda:0", backend=Backend.MARLIN)

    def test_marlin_hf_cache_serialization(self):
        model = GPTQModel.from_quantized(self.MODEL_ID, device="cuda:0", backend=Backend.MARLIN)
        self.assertTrue(model.quantize_config.format == FORMAT.MARLIN)

        model = GPTQModel.from_quantized(self.MODEL_ID, device="cuda:0", backend=Backend.MARLIN)
        self.assertTrue(model.quantize_config.format == FORMAT.MARLIN)

    def test_gptq_v1_to_v2_runtime_convert(self):
        model = GPTQModel.from_quantized(self.MODEL_ID, device="cuda:0")
        self.assertTrue(model.quantize_config.format == FORMAT.GPTQ_V2)

    def test_gptq_v1_serialization(self):
        model = GPTQModel.from_quantized(self.MODEL_ID, device="cuda:0")
        model.quantize_config.format = FORMAT.GPTQ

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save_quantized(tmpdir)

            with open(os.path.join(tmpdir, "quantize_config.json"), "r") as f:
                quantize_config = json.load(f)

            self.assertTrue(quantize_config[FORMAT_FIELD_JSON] == "gptq")
