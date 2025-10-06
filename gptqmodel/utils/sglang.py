# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import multiprocessing as mp

import torch
from transformers import AutoConfig


try:
    import sglang as sgl
    SGLANG_AVAILABLE = True
except ImportError:
    SGLANG_AVAILABLE = False

SGLANG_INSTALL_HINT = "sglang not installed. Please install via `pip install -U sglang`."

def load_model_by_sglang(
    model,
    trust_remote_code,
    **kwargs
):
    if not SGLANG_AVAILABLE:
        raise ValueError(SGLANG_INSTALL_HINT)

    mp.set_start_method('spawn')
    runtime = sgl.Runtime(
        model_path=model,
        **kwargs,
    )
    sgl.set_default_backend(runtime)
    hf_config = AutoConfig.from_pretrained(
        model, trust_remote_code=trust_remote_code
    )
    return runtime, hf_config

if SGLANG_AVAILABLE:
    @sgl.function
    def generate(s, prompt, **kwargs):
        s += prompt
        s += sgl.gen("result", **kwargs)
else:
    def generate(s, prompt, **kwargs):
        print(SGLANG_INSTALL_HINT)

@torch.inference_mode
def sglang_generate(
        model,
        **kwargs,
):
    if not SGLANG_AVAILABLE:
        raise ValueError(SGLANG_INSTALL_HINT)

    prompts = kwargs.pop("prompts", None)
    state = generate.run(
        prompt=prompts,
        **kwargs,
    )

    return state["result"]
