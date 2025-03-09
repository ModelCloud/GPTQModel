# -- do not touch
import os
import unittest

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch
from logbar import LogBar, LogBar
from gptqmodel import GPTQModel, BACKEND # noqa: E402

log = LogBar.shared()

class TestGroupSize(unittest.TestCase):
    def test_load_group_128(self):
        model = GPTQModel.load(
            "/monster/data/model/QQQ-Llama-3-8b-g128",
        )

        result = model.generate("Uncovering deep insights begins with")[0] # tokens
        log.info(f"Output: {model.tokenizer.decode(result)}") # string output
