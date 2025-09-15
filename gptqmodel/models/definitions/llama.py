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


class LlamaGPTQ(BaseGPTQModel):
    # Non-repeating layers at the root level: same level as `layers_node`
    # Excluding `layers_node`.
    base_modules = ["model.embed_tokens", "model.norm"]
    pre_lm_head_norm_module = "model.norm"
    embed_modules = ["model.embed_tokens", "model.rotary_emb"]

    # Below describes all the repeating layers in this transformer model
    # `model.layers` is a node/module that hold all the repeating layers. The parent node for all n-layers.
    layers_node = ["model.layers"]
    # Each repeating layer in `model.layers` is of type `LlamaDecoderLayer`
    layer_type = "LlamaDecoderLayer"

    # Full tree of quantizable modules
    # `#` str will match any number: useful for layers and moe indexing.
    # List[str] for serial linked nodes. List str are linear depth linked modules presented in a linear fashion with no divergence.
    # Dict{str: List[str] | Dict | Tuple[str]} for diverging nodes where a node splits into multiple paths/nodes.
    # Tuple(str) for final targeted modules/nodes: there are only strings representing the final targeted modules
    layers_modules_tree = [
        "model",
        "layers",
        "#",
        {
            "self_attn": ("k_proj", "v_proj", "q_proj", "o_proj"),
            "mlp": ("up_proj", "gate_proj", "down_proj"),
        }
    ]

    # TODO: full deprecation by gptqmodel v4.3
    # legacy definition (deprecated): migrate to layers_modules_tree
    # Inside each `LlamaDecoderLayer` layer are many internal modules
    # List them in the order executed in model forward() code
    # Many models have same execution order of: attention (q_k_v) projection, attention (output) projection, mlp (n) projections
    layer_modules = [
        ["input_layernorm!"],
        ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
        ["input_layernorm!"],
        ["self_attn.o_proj"],
        ["mlp.gate_proj", "mlp.up_proj",],
        ["mlp.down_proj"],
    ]


    def get_layers_for_scaling(self, module, input_feat, module_kwargs):
        nodes = []
        last_module = None  # most recent norm obj (from a '!...' block)
        last_module_root = None # self_attn.* has root == self_attn, mlp.* has root == mlp

        for block in self.layer_modules:
            not_quantized = all(name.endswith("!") for name in block)

            if not_quantized:
                # Remember the latest norm (use the last entry if multiple are present)
                last_module = get_attr(module, block[-1].strip("!"))
                continue

            # Normal execution subset
            subset = []
            for name in block:
                if not name.endswith("!"):
                    subset.append(get_attr(module, name))

            assert len(subset) > 0

            prev_op = last_module

            assert prev_op != None

            n = dict(
                    prev_op=prev_op,
                    layers=subset,
                    inp=input_feat[block[0]],
                    kwargs=module_kwargs
                )

            root_split = block[0].split(".", 2)
            if len(root_split) == 2:
                root = root_split[0]
                if root != last_module_root:
                    last_module_root = root
                    node["module2inspect"] = get_attr(module, root)


            nodes.append(n)

            # Update tracker to the LAST item of this block
            last_module = getattr(module, block[-1].strip("!"))

        print(f"DEBUG AWQ NODES: {nodes}")
        return nodes


    # def get_layers_for_scaling(self, module, input_feat, module_kwargs):
    #     layers = []
    #
    #     # attention input
    #     layers.append(
    #         dict(
    #             prev_op=module.input_layernorm,
    #             layers=[
    #                 module.self_attn.q_proj,
    #                 module.self_attn.k_proj,
    #                 module.self_attn.v_proj,
    #             ],
    #             inp=input_feat["self_attn.q_proj"],
    #             module2inspect=module.self_attn,
    #             kwargs=module_kwargs,
    #         )
    #     )
    #
    #     # attention out
    #     # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
    #     # Not GPA attention we need to skip o projection quant
    #     if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
    #         layers.append(
    #             dict(
    #                 prev_op=module.self_attn.v_proj,
    #                 layers=[module.self_attn.o_proj],
    #                 inp=input_feat["self_attn.o_proj"],
    #             )
    #         )
    #
    #     # linear 1
    #     layers.append(
    #         dict(
    #             prev_op=module.post_attention_layernorm,
    #             layers=[module.mlp.gate_proj, module.mlp.up_proj],
    #             inp=input_feat["mlp.gate_proj"],
    #             module2inspect=module.mlp,
    #         )
    #     )
    #
    #     # linear 2
    #     layers.append(
    #         dict(
    #             prev_op=module.mlp.up_proj,
    #             layers=[module.mlp.down_proj],
    #             inp=input_feat["mlp.down_proj"],
    #         )
    #     )
    #
    #     return layers