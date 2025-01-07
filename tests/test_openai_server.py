import os
import unittest

from gptqmodel import GPTQModel
import openai
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


class TestOpeniServer(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct-gptqmodel-4bit-vortext-v1"
        self.HOST = "127.0.0.1"
        self.PORT = 23900


    def test_openai_server(self):
        model = GPTQModel.load(self.MODEL_ID)
        model.serve(host=self.HOST, port=self.PORT, async_mode=True)

        time.sleep(5)

        client = openai.Client(base_url=f"http://{self.HOST}:{self.PORT}/v1", api_key="None")

        messages = [
            {"role": "user", "content": "1+1=?"},
        ]

        response = client.chat.completions.create(
            model=self.MODEL_ID,
            messages=messages,
            temperature=0,
        )
        result_text = response.choices[0].text

        client.get(f"http://{self.HOST}:{self.PORT}/shutdown", cast_to=str)

        self.assertEqual(result_text.strip(), "1 + 1 = 2")









