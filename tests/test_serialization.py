# -- do not touch
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

import json  # noqa: E402
import os  # noqa: E402
import tempfile  # noqa: E402
import unittest  # noqa: E402

from gptqmodel import BACKEND, GPTQModel  # noqa: E402
from gptqmodel.quantization import FORMAT, FORMAT_FIELD_JSON  # noqa: E402


class TestSerialization(unittest.TestCase):
    MODEL_ID = "/monster/data/model/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"

    def test_marlin_local_serialization(self):
        model = GPTQModel.load(self.MODEL_ID, device="cuda:0", backend=BACKEND.MARLIN)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(tmpdir)

            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "model.safetensors")))

            model = GPTQModel.load(tmpdir, device="cuda:0", backend=BACKEND.MARLIN)

    def test_gptq_v1_to_v2_runtime_convert(self):
        model = GPTQModel.load(self.MODEL_ID, device="cuda:0")
        self.assertEqual(model.quantize_config.runtime_format, FORMAT.GPTQ_V2)

    def test_gptq_v1_serialization(self):
        model = GPTQModel.load(self.MODEL_ID, device="cuda:0")
        model.quantize_config.format = FORMAT.GPTQ

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(tmpdir)

            with open(os.path.join(tmpdir, "quantize_config.json"), "r") as f:
                quantize_config = json.load(f)

            self.assertEqual(quantize_config[FORMAT_FIELD_JSON], "gptq")
