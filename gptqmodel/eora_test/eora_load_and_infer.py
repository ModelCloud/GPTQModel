import os

from gptqmodel import BACKEND, GPTQModel
from gptqmodel.adapter.adapter import Lora
from parameterized import parameterized


@parameterized.expand([
    (BACKEND.TORCH),
    (BACKEND.CUDA),
    (BACKEND.TRITON),
    (BACKEND.EXLLAMA_V1),
    # (BACKEND.EXLLAMA_V2), <-- adapter not working yet
    (BACKEND.MARLIN),
    # (BACKEND.IPEX), <-- not tested yet
    # (BACKEND.BITBLAS, <-- not tested yet
])
def test_load(backend: BACKEND):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"    
    quant_model_path = "/home/shihyangl/gptqmodel_save/Llama-3.2-1B-gptqmodel-4bit"
    lora_path = "/home/shihyangl/llama3.2-1b-4bit-group128-eora_test-rank128-arc/adapter_model.safetensors" #"sliuau/llama3.2-1b-4bit-group128-eora_test-rank128-arc/blob/main/adapter_model.safetensors" #"sliuau/llama3.2-1b-4bit-group128-eora_test-rank128-arc"

    adapter = Lora(path=lora_path, rank=128)

    model = GPTQModel.load(
        quant_model_path,
        adapter=adapter,
        backend=backend,
        device_map="auto",
    )

    # print(model)
    tokens = model.generate("Capital of France is")[0]
    result = model.tokenizer.decode(tokens)
    print(f"Result: {result}")
    assert "paris" in result.lower()



# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"    
# quant_model_path = "/home/shihyangl/gptqmodel_save/Llama-3.2-1B-gptqmodel-4bit"
# lora_path = "/home/shihyangl/llama3.2-1b-4bit-group128-eora_test-rank128-arc/adapter_model.safetensors" #"sliuau/llama3.2-1b-4bit-group128-eora_test-rank128-arc/blob/main/adapter_model.safetensors" #"sliuau/llama3.2-1b-4bit-group128-eora_test-rank128-arc"

# adapter = EoRA(lora_path=lora_path, rank=128)

# model = GPTQModel.load(
#     quant_model_path,
#     adapter=adapter,
#     backend=BACKEND.TORCH,
#     device_map="auto",
# )

# # print(model)
# tokens = model.generate("Capital of France is")[0]
# result = model.tokenizer.decode(tokens)
# print(f"Result: {result}")
# assert "paris" in result.lower()
