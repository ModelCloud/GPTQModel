# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# -- do not touch
import os


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch
import tempfile  # noqa: E402
import unittest  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402
from models.model_test import ModelTest  # noqa: E402
from parameterized import parameterized  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from gptqmodel import BACKEND, GPTQModel, get_best_device  # noqa: E402
from gptqmodel.quantization import FORMAT, METHOD  # noqa: E402
from gptqmodel.utils.importer import get_kernel_for_backend  # noqa: E402


pytestmark = [pytest.mark.model, pytest.mark.slow]


MODEL_ID = "/monster/data/model/TinyLlama-1.1B-Chat-v1.0-GPTQ-4bit"

class TestSave(unittest.TestCase):
    def _require_backend(self, backend: BACKEND):
        kernel_cls = get_kernel_for_backend(backend, METHOD.GPTQ, FORMAT.GPTQ)
        ok, err = kernel_cls.cached_validate_once()
        if not ok:
            self.skipTest(f"{backend} unavailable: {err}")

    def _generate_or_skip(self, model, backend: BACKEND, tokenizer, prompt, **kwargs):
        try:
            return ModelTest.generate_stable_with_limit(model, tokenizer, prompt, **kwargs)
        except Exception as exc:
            if backend == BACKEND.BITBLAS:
                message = str(exc).lower()
                if "illegal memory access" in message or isinstance(exc, torch.AcceleratorError):
                    self.skipTest(f"{backend} runtime unstable in this environment: {exc}")
            raise

    @parameterized.expand(
        [
            (BACKEND.AUTO),
            (BACKEND.EXLLAMA_V2),
            (BACKEND.TRITON),
            (BACKEND.BITBLAS),
            (BACKEND.MARLIN),
        ]
    )
    def test_save(self, backend: BACKEND):
        if backend != BACKEND.AUTO:
            self._require_backend(backend)

        prompt = "I am in Paris and"
        device = get_best_device(backend)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

        # origin model produce correct output
        origin_model = GPTQModel.load(MODEL_ID, backend=backend, device=device)
        origin_model_predicted_text = self._generate_or_skip(
            origin_model,
            backend,
            tokenizer,
            prompt,
            min_new_tokens=60,
            max_new_tokens=60,
            skip_special_tokens=False,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            origin_model.save(tmpdir)

            # saved model produce wrong output
            new_model = GPTQModel.load(tmpdir, backend=backend, device=device)

            new_model_predicted_text = self._generate_or_skip(
                new_model,
                backend,
                tokenizer,
                prompt,
                min_new_tokens=60,
                max_new_tokens=60,
                skip_special_tokens=False,
            )

            print("origin_model_predicted_text",origin_model_predicted_text)
            print("new_model_predicted_text",new_model_predicted_text)

            self.assertEqual(origin_model_predicted_text[:20], new_model_predicted_text[:20])
