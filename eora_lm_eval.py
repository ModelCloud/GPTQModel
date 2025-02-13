# -- do not touch
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

from gptqmodel import BACKEND, GPTQModel  # noqa: E402
from gptqmodel.adapter.adapter import Lora  # noqa: E402
from tests.models.model_test import ModelTest  # noqa: E402
from parameterized import parameterized  # noqa: E402
import unittest

class Test(ModelTest):
    NATIVE_MODEL_ID = "/home/shihyangl/gptqmodel_save/Llama-3.2-1B-gptqmodel-4bit"
    lora_path = "/home/shihyangl/llama3.2-1b-4bit-group128-eora-rank128-c4-v2/adapter_model.safetensors" #"sliuau/llama3.2-1b-4bit-group128-eora-rank128-arc/blob/main/adapter_model.safetensors" #"sliuau/llama3.2-1b-4bit-group128-eora-rank128-arc"

    NATIVE_ARC_CHALLENGE_ACC = 0.3567
    NATIVE_ARC_CHALLENGE_ACC_NORM = 0.3805
    QUANT_ARC_MAX_DELTA_FLOOR_PERCENT = 0.36

    @classmethod
    def setUpClass(cls):
        cls.adapter = Lora(path=cls.lora_path, rank=128)

    @parameterized.expand([
        BACKEND.TORCH,
        # BACKEND.CUDA,
        # BACKEND.TRITON,
        # BACKEND.EXLLAMA_V1,
        # (BACKEND.EXLLAMA_V2), <-- adapter not working yet
        # BACKEND.MARLIN,
        # (BACKEND.IPEX), <-- not tested yet
        # (BACKEND.BITBLAS, <-- not tested yet
    ])
    def test_load(self, backend: BACKEND):
        model = GPTQModel.load(
            self.NATIVE_MODEL_ID,
            adapter=self.adapter,
            backend=backend,
            device_map="auto",
        )

        # print(model)
        tokens = model.generate("Capital of France is")[0]
        result = model.tokenizer.decode(tokens)
        print(f"Result: {result}")
        assert "paris" in result.lower()

    def test_lm_eval_from_path(self):
        print("test_lm_eval_from_path")
        adapter = Lora(path=self.lora_path, rank=128)
        task_results = self.lm_eval(None, extra_args={"adapter": adapter.to_dict()})
        self.check_results(task_results)

    def test_lm_eval_from_model(self):
        print("test_lm_eval_from_model")
        model = GPTQModel.load(
            self.NATIVE_MODEL_ID,
            adapter=self.adapter,
            backend=BACKEND.TRITON,
        )
        task_results = self.lm_eval(model)
        self.check_results(task_results)


if __name__ == '__main__':
    unittest.main()
