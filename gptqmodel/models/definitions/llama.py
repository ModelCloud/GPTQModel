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

from ..base import BaseGPTQModel, classproperty

from collections import defaultdict

def build_layer_modules(tree):
    """
    tree format:
      [<model_name>, <submodule>, "#", { parent_module: ( "child[:!][:grp]", ... ), ... }]
    Rules:
      - ':!' means participates in inference but is NOT quantized; keep this marker in output.
      - ':<digit>' means grouping; children with the same group id are emitted in the same block.
      - Both can appear together, e.g. 'module_name:!:2'.
    Output:
      _layer_modules = [ [items...], [items...], ... ]
    """
    # Be lenient: just require a dict as the 4th element.
    if not (isinstance(tree, list) and len(tree) >= 4 and isinstance(tree[3], dict)):
        raise ValueError("layers_modules_tree must be ['model','layers','#',{...}] (4th element a dict)")

    mapping = tree[3]
    out_blocks = []

    for parent, entries in mapping.items():
        groups = defaultdict(list)

        for ent in entries:
            parts = ent.split(':')
            child = parts[0]

            flags = parts[1:]
            has_bang = ('!' in flags)
            # first numeric tag is the group id; default 0
            grp = next((int(p) for p in flags if p.isdigit()), 0)

            groups[grp].append((child, has_bang))

        # Emit per-group, skipping pure-:! blocks (norm-only), but
        # preserving :! markers on mixed blocks if they ever occur.
        for g in sorted(groups):
            items = groups[g]
            # if every entry is :!, skip this block (matches your expected output)
            if all(has_bang for _, has_bang in items):
                continue

            block = []
            for child, has_bang in items:
                full = child if child == parent else f"{parent}.{child}"
                if has_bang:
                    full += ":!"
                block.append(full)

            out_blocks.append(block)

    return out_blocks

def generate_layers_modules_tree_simple(node):
    """
    Recursively walk a nested list/dict structure and:
      1. Drop dict entries where *all* values are ':!' flagged.
      2. Remove ':!' and ':<digit>' markers from strings.
    """

    # If it's a list, recurse into each element
    if isinstance(node, list):
        return [generate_layers_modules_tree_simple(x) for x in node]

    # If it's a dict, process each key -> value
    if isinstance(node, dict):
        new_dict = {}
        for k, v in node.items():
            # Expand tuple-of-strings blocks (special handling)
            if isinstance(v, (tuple, list)) and all(isinstance(x, str) for x in v):
                # Rule 1: check if ALL entries are :!
                if all(any(p == "!" for p in x.split(":")[1:]) for x in v):
                    continue  # skip this parent entirely

                # Rule 2: strip :! and :digit markers
                cleaned = tuple(x.split(":")[0] for x in v)
                new_dict[k] = cleaned
            else:
                # Recurse deeper
                new_dict[k] = convert_tree(v)
        return new_dict

    # If it's a plain string (unlikely here), strip markers
    if isinstance(node, str):
        return node.split(":")[0]

    # For other types, return as-is
    return node


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

    _layers_modules_tree = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": ("q_proj:0", "k_proj:0", "v_proj:0", "o_proj:1"),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("up_proj:0", "gate_proj:0", "down_proj:1"),
        }
    ]

    layers_modules_tree = generate_layers_modules_tree_simple(_layers_modules_tree)
    print(f"layers_modules_tree: {layers_modules_tree}")

    layer_modules = build_layer_modules(layers_modules_tree)
    print(f"layers_modules: {layer_modules}")


