# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import pytest
import torch
from awq_test_utils import run_inference_only_generation_test

from gptqmodel import BACKEND


MODEL_ID = "ModelCloud/opt-125m-llm-awq"


def _cached(model_id: str) -> bool:
    from huggingface_hub import try_to_load_from_cache

    return isinstance(try_to_load_from_cache(model_id, "config.json"), str)


pytestmark = [
    pytest.mark.model,
    pytest.mark.slow,
    # The shared helper loads onto `cuda` and fetches the Hub checkpoint.
    pytest.mark.skipif(not torch.cuda.is_available(), reason="AWQ inference test loads onto CUDA"),
    pytest.mark.skipif(not _cached(MODEL_ID), reason=f"{MODEL_ID} is not in the local Hugging Face cache"),
]


def test_inference_quantized_by_llm_awq():
    run_inference_only_generation_test(
        MODEL_ID,  # this quantized by llm-awq
        backend=BACKEND.AUTO,
        max_new_tokens=512,
        extra_terms=("food", "market", "country"),
    )
