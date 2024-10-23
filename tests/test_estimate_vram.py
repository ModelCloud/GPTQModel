# -- do not touch
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

import tempfile  # noqa: E402
import unittest  # noqa: E402
from gptqmodel import BACKEND, GPTQModel  # noqa: E402
from gptqmodel.utils import get_vram

class TestEstimateVram(unittest.TestCase):
    NATIVE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    def test_estimate_vram(self):
        model = GPTQModel.from_quantized(
            self.NATIVE_MODEL_ID,
            device_map="auto",
        )

        total_size, all_layers = get_vram(model)
        print(f"{self.NATIVE_MODEL_ID} estimate vram : {total_size}")
        for layer in all_layers:
            layer_name, layer_size = layer
            print(f"Layer {layer_name}: {layer_size}")
        del model
        assert total_size > 10