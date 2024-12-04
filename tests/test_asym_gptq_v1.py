# -- do not touch
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch
import tempfile  # noqa: E402
from parameterized import parameterized  # noqa: E402

from gptqmodel import QuantizeConfig, GPTQModel  # noqa: E402
from gptqmodel.quantization import FORMAT  # noqa: E402

from models.model_test import ModelTest  # noqa: E402


class Test(ModelTest):
    NATIVE_MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct"  # "meta-llama/Llama-3.2-1B-Instruct"
    QUANT_FORMAT = FORMAT.GPTQ
    QUANT_ARC_MAX_NEGATIVE_DELTA = 0.2

    @classmethod
    def setUpClass(self):
        self.tokenizer = self.load_tokenizer(self.NATIVE_MODEL_ID)
        self.datas = self.load_dataset(self.tokenizer)

    @parameterized.expand([True, False])
    def test(self, sym: bool):
        if sym:
            self.NATIVE_ARC_CHALLENGE_ACC = 0.2269
            self.NATIVE_ARC_CHALLENGE_ACC_NORM = 0.2269
        else:
            self.NATIVE_ARC_CHALLENGE_ACC = 0.2747
            self.NATIVE_ARC_CHALLENGE_ACC_NORM = 0.2935
        quantize_config = QuantizeConfig(
            format=self.QUANT_FORMAT,
            desc_act=self.DESC_ACT,
            sym=sym,
        )
        model = GPTQModel.load(
            self.NATIVE_MODEL_ID,
            quantize_config=quantize_config,
            device_map="auto",
        )

        model.quantize(self.datas)

        with (tempfile.TemporaryDirectory()) as tmpdirname:
            model.save(tmpdirname)
            self.tokenizer.save_pretrained(tmpdirname)
            model, tokenizer = self.loadQuantModel(tmpdirname)

            task_results = self.lm_eval(model=model)
            self.check_results(task_results)
