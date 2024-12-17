# -- do not touch
import os
import tempfile

from gptqmodel.models._const import DEVICE
from gptqmodel.utils.importer import backend_dict

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

import torch  # noqa: E402
from gptqmodel import BACKEND, GPTQModel, QuantizeConfig  # noqa: E402
from models.model_test import ModelTest  # noqa: E402

GENERATE_EVAL_SIZE = 100


class TestsIPEX(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/opt-125m"  # "facebook/opt-125m"

    def test(self):
        origin_model = GPTQModel.load(
            self.NATIVE_MODEL_ID,
            quantize_config=QuantizeConfig(),
            backend=BACKEND.IPEX,
            device=DEVICE.XPU,
        )
        tokenizer = self.load_tokenizer(self.NATIVE_MODEL_ID)
        calibration_dataset = self.load_dataset(tokenizer)
        origin_model.quantize(calibration_dataset)
        with tempfile.TemporaryDirectory() as tmpdir:
          origin_model.save(tmpdir)

          model = GPTQModel.load(
              tmpdir,
              backend=BACKEND.IPEX,
              device=DEVICE.XPU,
          )
          generate_str = tokenizer.decode(model.generate(**tokenizer("The capital of France is is", return_tensors="pt").to(model.device), max_new_tokens=2)[0])

          print(f"generate_str: {generate_str}")

          self.assertIn("paris", generate_str.lower())
