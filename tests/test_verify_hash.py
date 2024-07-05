import unittest

from gptqmodel import BACKEND, GPTQModel


class TestVerifyHashFunction(unittest.TestCase):
    MODEL_ID = "LnL-AI/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"
    EXPECTED_MD5_HASH = "md5:7725c72bc217bcb57b3f1f31d452d871"
    EXPECTED_SHA256_HASH = "sha256:2680bb4d5c977ee54f25dae584665641ea887e7bd8e8d7197ce8ffd310e93f2f"


    def test_verify_md5_hash_function(self):
        # Load the model with MD5 verify_hash parameter
        model = GPTQModel.from_quantized(self.MODEL_ID, device="cuda:0", backend=BACKEND.MARLIN,
                                         verify_hash=self.EXPECTED_MD5_HASH)
        self.assertIsNotNone(model)

    def test_verify_sha256_hash_function(self):
        # Load the model with SHA-256 verify_hash parameter
        model = GPTQModel.from_quantized(self.MODEL_ID, device="cuda:0", backend=BACKEND.MARLIN,
                                         verify_hash=self.EXPECTED_SHA256_HASH)
        # Add additional checks to ensure the model is loaded correctly
        self.assertIsNotNone(model)

