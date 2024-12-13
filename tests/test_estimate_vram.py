# -- do not touch
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch
import unittest  # noqa: E402

from gptqmodel import GPTQModel  # noqa: E402
from gptqmodel.quantization import QuantizeConfig  # noqa: E402
from gptqmodel.utils import get_vram  # noqa: E402


class TestEstimateVram(unittest.TestCase):
    NATIVE_MODEL_ID = "/monster/data/model/TinyLlama-1.1B-Chat-v1.0" # "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    def test_estimate_vram(self):
        quantize_config = QuantizeConfig(
            bits=4,
            group_size=128,
        )
        model = GPTQModel.load(
            self.NATIVE_MODEL_ID,
            quantize_config=quantize_config,
        )

        total_size, all_layers = get_vram(model)
        print(f"{self.NATIVE_MODEL_ID} estimate vram : {total_size}")
        for layer in all_layers:
            layer_name, layer_size = layer
            print(f"Layer {layer_name}: {layer_size}")
        del model

        assert total_size == "2.05 GB"
