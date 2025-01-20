import os.path
import tempfile
import unittest

import torch.cuda
from datasets import load_dataset
from huggingface_hub import hf_api
from safetensors.torch import load_file

from gptqmodel import GPTQModel, QuantizeConfig
from gptqmodel.utils.tensor import tensor_parameters


class TestsParameterCount(unittest.TestCase):
    LLAMA_3_2_1B_PARAMETER_COUNT = 1235814400

    # ModelCloud/Llama-3.2-1B-Instruct-gptqmodel-4bit-vortex-v1 incorrectly saves lm_head.weight,
    # and the number of calculated parameters will be larger.
    # The latest code has fixed this bug.
    LLAMA_3_2_1B_VORTEX_V1_PARAMETER_COUNT = 1498482688

    def test_parameter_count(self):
        import os.path

        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        from gptqmodel import QuantizeConfig
        from gptqmodel.utils.tensor import tensor_parameters

        model_id = "/monster/data/model/Llama-3.2-1B-Instruct-gptqmodel-4bit-vortex-v1"
        if os.path.isdir(model_id):
            file_path = os.path.join(model_id, "model.safetensors")
            config_path = os.path.join(model_id, "config.json")
        else:
            file_path = hf_hub_download(model_id, filename="model.safetensors")
            config_path = hf_hub_download(model_id, filename="config.json")
        safetensors_obj = load_file(file_path)
        quantize_config = QuantizeConfig.from_pretrained(os.path.dirname(config_path))

        total_params = 0

        for name, tensor in safetensors_obj.items():
            param_count = tensor_parameters(name, tensor.shape, bits=quantize_config.bits)
            total_params += param_count

        print(f"total_params: {total_params / 1e9} B")

        self.assertEqual(total_params, self.LLAMA_3_2_1B_VORTEX_V1_PARAMETER_COUNT)

    def test_parameter_count_with_quant(self):
        model_id = "/monster/data/model/Llama-3.2-1B-Instruct"  # meta-llama/Llama-3.2-1B-Instruct

        calibration_dataset = load_dataset(
            "allenai/c4",
            data_files="en/c4-train.00001-of-01024.json.gz",
            split="train"
        ).select(range(128))["text"]

        quant_config = QuantizeConfig(bits=4, group_size=128)

        model = GPTQModel.load(model_id, quant_config)

        # increase `batch_size` to match gpu/vram specs to speed up quantization
        model.quantize(calibration_dataset, batch_size=8)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save(tmp_dir)
            del model
            torch.cuda.empty_cache()

            safetensors_obj = load_file(os.path.join(tmp_dir, "model.safetensors"))
            total_params = 0
            for name, tensor in safetensors_obj.items():
                param_count = tensor_parameters(name, tensor.shape, bits=quant_config.bits)
                total_params += param_count

            print(f"total_params: {total_params / 1e9} B")

            self.assertEqual(total_params, self.LLAMA_3_2_1B_PARAMETER_COUNT)
