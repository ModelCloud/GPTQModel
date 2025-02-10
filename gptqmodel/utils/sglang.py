# Copyright 2024-2025 ModelCloud.ai
# Copyright 2024-2025 qubitium@modelcloud.ai
# Contact: qubitium@modelcloud.ai, x.com/qubitium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
