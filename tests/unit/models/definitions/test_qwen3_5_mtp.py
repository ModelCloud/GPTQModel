from gptqmodel.models.definitions.qwen3_5 import Qwen3_5QModel
from gptqmodel.models.definitions.qwen3_5_text import Qwen3_5TextQModel


def test_qwen3_5_preserves_mtp_out_of_model_tensors():
    assert Qwen3_5QModel.out_of_model_tensors == {"prefixes": ["mtp"]}


def test_qwen3_5_text_preserves_mtp_out_of_model_tensors():
    assert Qwen3_5TextQModel.out_of_model_tensors == {"prefixes": ["mtp"]}
