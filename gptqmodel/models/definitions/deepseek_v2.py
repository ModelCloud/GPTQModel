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

from .._const import EXPERT_INDEX_PLACEHOLDER
from ..base import BaseGPTQModel


# Both DeepSeek-v2 and DeepSeek-v2-lite are supported in this model def
class DeepSeekV2GPTQ(BaseGPTQModel):
    # deepseek_v2 requires custom model code
    require_trust_remote_code = True

    # allow dynamic expert index for layer_modules so we don't need to write out 64 layers here
    # config.num_experts contains the actual expert count used for index
    dynamic_expert_index = "n_routed_experts"

    base_modules = ["model.embed_tokens", "model.norm"]
    pre_lm_head_norm_module = "model.norm"

    layers_node = "model.layers"
    layer_type = "DeepseekV2DecoderLayer"

    # DeepSeek V2-Lite uses dynamic modules based on lora(rank):
    # https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/main/modeling_deepseek.py#L712
    layer_modules_strict = False

    # DeepSeek-V2 uses 160 experts, v2-lite is auto-switched during __init__
    layer_modules = [
        # DeepSeek-V2 and DeepSeek-V2-Lite use same model_type, but different self_attn
        # so we provide different layer_modules usage.
        # DeepSeek-V2-Lite usage
        #["self_attn.q_proj", "self_attn.kv_a_proj_with_mqa", "self_attn.kv_b_proj"],

        # DeepSeek-V2 usage, included in layer 0-59
        #["self_attn.q_a_proj", "self_attn.q_b_proj", "self_attn.kv_a_proj_with_mqa", "self_attn.kv_b_proj"],

        # merged v2-lite and v2
        ["self_attn.q_a_proj", "self_attn.q_b_proj", "self_attn.q_proj", "self_attn.kv_a_proj_with_mqa", "self_attn.kv_b_proj"],

        ["self_attn.o_proj"],

        # included in layer 0
        ["mlp.gate_proj", "mlp.up_proj"],
        ["mlp.down_proj"],

        # included in layer 1-59, uses dynamic_expert_index
        [f"mlp.experts.{EXPERT_INDEX_PLACEHOLDER}.gate_proj", f"mlp.experts.{EXPERT_INDEX_PLACEHOLDER}.up_proj"],
        [f"mlp.experts.{EXPERT_INDEX_PLACEHOLDER}.down_proj"],

        # included in layer 1-59
        ["mlp.shared_experts.gate_proj", "mlp.shared_experts.up_proj"],
        ["mlp.shared_experts.down_proj"],
    ]
