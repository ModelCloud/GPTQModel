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


from ..base import BaseGPTQModel


class Phi4MMGPTQ(BaseGPTQModel):
    base_modules = ["model.embed_tokens", "model.embed_tokens_extend", "model.norm"]
    pre_lm_head_norm_module = "model.norm"

    layers_node = "model.layers"
    layer_type = "Phi4MMDecoderLayer"

    # text modules only
    layer_modules = [
        ["self_attn.qkv_proj.base_layer"],
        ["self_attn.o_proj.base_layer"],
        ["mlp.gate_up_proj.base_layer"],
        ["mlp.down_proj.base_layer"],
    ]

    require_monkeypatch = True

    def monkey_patch(self):
        if not self.quantized:
            original_forward = self.model.forward

            # patch so input_mode is default to 0 (InputMode.LANGUAGE) if not passed
            # phi4mm default is None which causes quant error as it expects it to be always passed
            def patched_forward(self, **kwargs):
                if "input_mode" not in kwargs:
                    kwargs["input_mode"] = 0
                return original_forward(**kwargs)

            # bind forward to instance
            self.model.forward = patched_forward.__get__(self.model, type(self.model))

__all__ = ["Phi4MMGPTQ"]
