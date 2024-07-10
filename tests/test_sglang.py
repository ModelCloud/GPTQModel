import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch
import asyncio
import unittest  # noqa: E402

from gptqmodel import BACKEND, GPTQModel  # noqa: E402


class TestLoadSglang(unittest.TestCase):
    MODEL_ID = "LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"
    prompt = "Hello, my name is"

    sampling_params = {
        "temperature": 0.8,
        "top_p": 0.95,
    }

    async def async_test_load_sglang(self):
        model = GPTQModel.from_quantized(
            self.MODEL_ID,
            device="cuda:0",
            backend=BACKEND.SGLANG,
            disable_flashinfer=True,
        )
        self.assertTrue(model is not None)
        stream = await model.generate(
            prompts=self.prompt,
            sampling_params=self.sampling_params,
        )
        buffer = []
        async for output in stream:
            buffer.append(output)
        print("".join(buffer))

        stream_params = await model.generate(
            prompts=self.prompt,
            temperature=0.8,
            top_p=0.95,
        )
        buffer_params = []
        async for output_params in stream_params:
            buffer_params.append(output_params)
        print("".join(buffer_params))

        model.shutdown()
        del model



    def test_load_sglang(self):
        asyncio.run(self.async_test_load_sglang())



