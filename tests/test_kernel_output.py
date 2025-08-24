import unittest

import torch
from gptqmodel import BACKEND, GPTQModel
from gptqmodel.adapter.adapter import Adapter, AdapterCache, Lora
from gptqmodel.nn_modules.qlinear.exllama_eora import ExllamaEoraQuantLinear
from gptqmodel.nn_modules.qlinear.exllamav2 import ExllamaV2QuantLinear
from gptqmodel.nn_modules.qlinear.marlin import MarlinQuantLinear
from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear
from gptqmodel.nn_modules.qlinear.tritonv2 import TritonV2QuantLinear
from gptqmodel.utils.model import find_modules
from logbar import LogBar
from parameterized import parameterized
from torch import Tensor

log = LogBar.shared()

DEVICE = torch.device("cuda:0")

class Data:
    def __init__(self):
        self.m = 1
        self.k = -1
        self.x = []  # random X input of shape (m, k)

class TestKernelOutput(unittest.TestCase):
    # model_path = "sliuau/llama3.2-1b-4bit-group128" # hf "sliuau/llama3.2-1b-4bit-group128"
    model_path = "sliuau/Llama-3.2-3B_4bits_128group_size"
    target_qliner_map = {
        # BACKEND.EXLLAMA_V1: ExllamaQuantLinear,
        BACKEND.EXLLAMA_EORA: ExllamaEoraQuantLinear,
        BACKEND.EXLLAMA_V2: ExllamaV2QuantLinear,
        BACKEND.TRITON: TritonV2QuantLinear,
        BACKEND.TORCH: TorchQuantLinear,
        # BACKEND.TORCH_FUSED: TorchFusedQuantLinear,
        # BACKEND.BITBLAS: BitBLASQuantLinear,
        # BACKEND.IPEX: IPEXQuantLinear,
        BACKEND.MARLIN: MarlinQuantLinear,
        # BACKEND.MARLIN_FP16: MarlinQuantLinear,
    }

    target = 'model.layers.6.self_attn.v_proj'
    m = [   # tuple is dim_0 size and num_sampes for each dim_0
            (1, 256),
            (16, 128),
            (32, 64),
            (64, 32),
            (128, 16),
        ]

    # sum all the second tuple value for total sample size
    random_input_sample_size = sum(t[1] for t in m)


    @classmethod
    def setUpClass(cls):
        # lora_path = "sliuau/llama3.2-1b-4bit-group128-eora-rank128-arc"  # adapter_model.safetensors
        lora_path = "sliuau/llama-3.2-3b_4bits_128group_size_eora_rank64_mmlu_c4"
        # hf "sliuau-llama3.2-1b-4bit-group128/llama3.2-1b-4bit-group128-eora-rank128-arc/"

        test_dtypes = [torch.float16, torch.bfloat16]
        cls.data = {} # key is dtype, v is Data()

        for dtype in test_dtypes:
            data = Data()

            # map data to dtype
            cls.data[dtype] = data

            data.adapter = Lora(
                rank=64,
                path=lora_path)

            data.adapter.post_init(cls.target, device=DEVICE) # trigger adapter weight load from disk
            data.k = data.adapter.lora_A.shape[0]

            for _ in log.pb(cls.random_input_sample_size).title("Generate Random Inputs"):
                for dim_0, samples in cls.m:

                    for _ in range(samples):
                        inputs = torch.rand((dim_0, data.k), device=DEVICE, dtype=dtype)
                        data.x.append(inputs)

            AdapterCache.reset() # allow next load to complete since we are hacking to get consume only 1 lora module

            # TORCH as reference output
            data.torch_kernel_out = cls.forward(cls, backend=BACKEND.TORCH, dtype=dtype)
            data.torch_kernel_out_with_lora = cls.forward(cls, backend=BACKEND.TORCH, dtype=dtype, adapter=data.adapter)


    def forward(self, backend: BACKEND, dtype: torch.dtype, adapter: Adapter = None):
        model = GPTQModel.load(self.model_path, backend=backend, adapter=adapter, torch_dtype=dtype)

        target_qlinear_cls = self.target_qliner_map[backend]

        modules = find_modules(model.model, layers=[target_qlinear_cls])
        result = []
        for name, module in modules.items():
            if name == self.target:
                data = self.data[dtype]
                for i in log.pb(self.random_input_sample_size).title("Forward Pass on Random Input"):
                    assert data.x[i].dtype == dtype
                    result.append(module(data.x[i]))
                break

        assert result is not None

        del module
        del model
        torch.cuda.empty_cache()

        return result

    def assert_on_mismatch(self, a: Tensor, b: Tensor, atol):
        torch.testing.assert_close(a, b, rtol=0.15, atol=atol)
        #torch.allclose(a, b, rtol=0.15, atol=atol)

    @parameterized.expand([
        (BACKEND.TORCH, torch.float16, 0.0000),
        # (BACKEND.TORCH_FUSED, torch.float16, 0.0000),
        (BACKEND.TRITON, torch.float16, 0.00001),
        # (BACKEND.EXLLAMA_V1, torch.float16, 0.0050),
        (BACKEND.EXLLAMA_V2, torch.float16, 0.0068),
        (BACKEND.MARLIN, torch.float16, 0.00035),
        # (BACKEND.MARLIN_FP16, torch.float16, 0.0035),
        (BACKEND.EXLLAMA_EORA, torch.float16, 0.0025),
    ])
    def test_kernel_float16(self, backend: BACKEND,  dtype: torch.dtype, a_tolerance: float):
        out = self.forward(backend=backend, dtype=dtype)

        # torch as ref
        pb = log.pb(len(out)).title(f"Actual Kernel Output With Lora {dtype}").manual()
        for i in pb:
            data = self.data[dtype]
            pb.subtitle(f"backed = `{backend}`").draw()
            try:
                self.assert_on_mismatch(data.torch_kernel_out[i], out[i],
                                        a_tolerance)  # use torch as reference
            except AssertionError:
                log.error(
                    f"Torch with Lora output: dtype = `{dtype}`, backed = `{BACKEND.TORCH}`, i = `{i}`, {data.torch_kernel_out[i][:10]}")
                log.error(
                    f"Actual with Lora output: dtype = `{dtype}`, backed = `{backend}`, i = `{i}`, {out[i][:10]}")
                raise AssertionError

    @parameterized.expand([
        (BACKEND.TORCH, torch.bfloat16, 0.0000),
        # (BACKEND.TORCH_FUSED, torch.bfloat16, 0.0000),
        (BACKEND.TRITON, torch.bfloat16, 0.00001),
        # (BACKEND.EXLLAMA_V1, torch.bfloat16, 0.0064),
        (BACKEND.EXLLAMA_V2, torch.bfloat16, 0.0054),
        (BACKEND.MARLIN, torch.bfloat16, 0.0031),
        # (BACKEND.MARLIN_FP16, torch.bfloat16, 0.012),
        # (BACKEND.EXLLAMA_EORA, torch.bfloat16, 0.0031), TODO FIX, abnormal output when Exllama Eora kernel is using bfloat16
    ])
    def test_kernel_bfloat16(self, backend: BACKEND, dtype: torch.dtype, a_tolerance: float):
        out = self.forward(backend=backend, dtype=dtype)

        # torch as ref
        pb = log.pb(len(out)).title(f"Actual Kernel Output With Lora {dtype}").manual()
        for i in pb:
            data = self.data[dtype]
            pb.subtitle(f"backed = `{backend}`").draw()
            try:
                self.assert_on_mismatch(data.torch_kernel_out[i], out[i],
                                        a_tolerance)  # use torch as reference
            except AssertionError:
                log.error(
                    f"Torch with Lora output: dtype = `{dtype}`, backed = `{BACKEND.TORCH}`, i = `{i}`, {data.torch_kernel_out[i][:10]}")
                log.error(
                    f"Actual with Lora output: dtype = `{dtype}`, backed = `{backend}`, i = `{i}`, {out[i][:10]}")
                raise AssertionError

    @parameterized.expand([
        (BACKEND.TORCH, torch.float16, 0.0000),
        # (BACKEND.TORCH_FUSED, torch.float16, 0.0000),
        (BACKEND.TRITON, torch.float16, 0.00001),
        # (BACKEND.EXLLAMA_V1, torch.float16, 0.0054),
        (BACKEND.EXLLAMA_V2, torch.float16, 0.0065),
        (BACKEND.MARLIN, torch.float16, 0.00035),
        # (BACKEND.MARLIN_FP16, torch.float16, 0.0035),
        (BACKEND.EXLLAMA_EORA, torch.float16, 0.0020)
    ])
    def test_kernel_float16_with_lora(self, backend: BACKEND, dtype: torch.dtype, a_tolerance: float):
        data = self.data[dtype]
        out = self.forward(backend=backend, dtype=dtype, adapter=data.adapter)

        # torch as ref
        pb = log.pb(len(out)).title(f"Actual Kernel Output With Lora {dtype}").manual()
        for i in pb:
            pb.subtitle(f"backed = `{backend}`").draw()
            try:
                self.assert_on_mismatch(data.torch_kernel_out_with_lora[i], out[i], a_tolerance)  # use torch as reference
            except AssertionError:
                log.error(f"Torch with Lora output: backed = dtype = `{dtype}`, `{backend}`, i = `{i}`, {data.torch_kernel_out_with_lora[i][:10]}")
                raise AssertionError


    @parameterized.expand([
        (BACKEND.TORCH, torch.bfloat16, 0.0000),
        # (BACKEND.TORCH_FUSED, torch.bfloat16, 0.0000),
        (BACKEND.TRITON, torch.bfloat16, 0.00001),
        # (BACKEND.EXLLAMA_V1, torch.bfloat16, 0.0062),
        (BACKEND.EXLLAMA_V2, torch.bfloat16, 0.0059),
        (BACKEND.MARLIN, torch.bfloat16, 0.0033),
        # (BACKEND.MARLIN_FP16, torch.bfloat16, 0.011),
        # (BACKEND.EXLLAMA_EORA, torch.bfloat16, 0.0014)  TODO FIX, abnormal output when Exllama Eora kernel is using bfloat16
    ])
    def test_kernel_bfloat16_with_lora(self, backend: BACKEND, dtype: torch.dtype, a_tolerance: float):
        data = self.data[dtype]
        out = self.forward(backend=backend, dtype=dtype, adapter=data.adapter)

        # torch as ref
        pb = log.pb(len(out)).title(f"Actual Kernel Output With Lora {dtype}").manual()
        for i in pb:
            pb.subtitle(f"backed = `{backend}`").draw()
            try:
                self.assert_on_mismatch(data.torch_kernel_out_with_lora[i], out[i], a_tolerance)  # use torch as reference
            except AssertionError:
                log.error(f"Torch with Lora output: dtype = `{dtype}`, backed = `{backend}`, i = `{i}`, {data.torch_kernel_out_with_lora[i][:10]}")
                raise AssertionError
