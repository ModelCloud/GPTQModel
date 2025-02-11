import os
os.environ["GPTQMODEL_USE_MODELSCOPE"] = "True"
import sys
import subprocess  # noqa: E402
import importlib.util  # noqa: E402

if importlib.util.find_spec("modelscope") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "modelscope>=1.23"])

from models.model_test import ModelTest  # noqa: E402
from gptqmodel import GPTQModel


class TestLoadModelscope(ModelTest):

    @classmethod
    def setUpClass(self):
        self.MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4"

    def test_load_modelscope(self):
        model = GPTQModel.load(self.MODEL_ID)

        result = model.generate("Hello")[0]
        assert(model.tokenizer.decode(result))

        del model