# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import os
import socket
import unittest

import openai
import pytest

from gptqmodel import GPTQModel


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

pytestmark = [pytest.mark.model, pytest.mark.slow]

class TestOpeniServer(unittest.TestCase):
    @staticmethod
    def _pick_free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]

    @classmethod
    def setUpClass(cls):
        cls.MODEL_ID = "/monster/data/model/Llama-3.2-1B-Instruct-gptqmodel-4bit-vortex-v1"
        cls.HOST = "127.0.0.1"
        cls.PORT = cls._pick_free_port()
        cls.model = GPTQModel.load(cls.MODEL_ID)

    @classmethod
    def tearDownClass(cls):
        try:
            cls.model.serve_shutdown()
        except Exception as exc:
            # Shutdown is best-effort here; surface failures without masking the test result.
            print(f"serve_shutdown failed during tearDownClass: {exc}")


    def test_openai_server(self):
        self.model.serve(host=self.HOST, port=self.PORT, async_mode=True)
        self.model.serve_wait_until_ready()
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
        self.assertEqual(result_text.strip(), "1 + 1 = 2")
        self.model.serve_shutdown()
