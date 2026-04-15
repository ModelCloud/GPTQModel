# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from packaging.version import Version
from transformers import __version__ as TRANSFORMERS_VERSION

from gptqmodel.models import definitions


def test_public_model_definition_exports():
    expected = [
        "BailingMoeQModel",
        "GLM4MoEGPTQ",
        "GPTOSSGPTQ",
        "Gemma3ForConditionalGenerationGPTQ",
        "Gemma4ForConditionalGenerationGPTQ",
        "Gemma4TextQModel",
        "GraniteMoeHybridQModel",
        "LFM2MoeQModel",
        "LLaDA2MoeQModel",
        "MiniCPMGPTQ",
        "OlmoeGPTQ",
        "Ovis2QModel",
        "Phi4MMGPTQ",
        "PhiMoEGPTQForCausalLM",
        "Qwen2_5_OmniGPTQ",
        "Qwen3NextGPTQ",
    ]

    for name in expected:
        assert hasattr(definitions, name), f"missing export: {name}"


def test_qwen3_5_exports_follow_transformers_support():
    supported = Version(TRANSFORMERS_VERSION) >= Version("5.2.0")
    if supported:
        assert definitions.Qwen3_5QModel is not None
        assert definitions.Qwen3_5_MoeQModel is not None
    else:
        assert definitions.Qwen3_5QModel is None
        assert definitions.Qwen3_5_MoeQModel is None
