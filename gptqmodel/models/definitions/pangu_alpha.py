# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import torch

from ...utils.model import move_to
from ..base import BaseQModel


class PanguAlphaQModel(BaseQModel):
    pre_lm_head_norm_module = "transformer.ln_f"

    # Full tree of quantizable modules
    # `#` str will match any number: useful for layers and moe indexing.
    # List[str] for serial linked nodes. List str are linear depth linked modules presented in a linear fashion with no divergence.
    # Dict{str: List[str] | Dict | Tuple[str]} for diverging nodes where a node splits into multiple paths/nodes.
    # Tuple(str) for final targeted modules/nodes: there are only strings representing the final targeted modules
    module_tree = [
        "transformer",
        "h",
        "#",
        {
            "ln_1": ("ln_1:!",),
            "attn": ("q_proj:0", "k_proj:0", "v_proj:0", "c_proj:1"),
            "ln_2": ("ln_2:!",),
            # `GPTPanguMLP.forward` is `c_proj(act(c_fc(x)))` with GELU, which is
            # not positively homogeneous: folding a scale into `c_fc` changes the
            # MLP output (4.5e-2 on a tiny model).
            "mlp": ("c_fc:0:nosrc", "c_proj:1"),
        }
    ]

    def awq_get_modules_for_scaling(self, module, input_feat, module_kwargs):
        """Drop the top-query layer's `q_proj` from the `ln_1` scaling group.

        ``GPTPanguModel.forward`` passes ``custom_query = wqe(position_ids)`` to
        the last block, so that block's ``q_proj`` does not read ``ln_1``'s
        output. Scaling it with the ``ln_1`` group changes the attention output
        (q is multiplied by the scale while its input is not divided).
        """

        nodes = super().awq_get_modules_for_scaling(module, input_feat, module_kwargs)
        transformer = getattr(self.model, "transformer", None)
        blocks = getattr(transformer, "h", None)
        if not blocks or blocks[-1] is not module:
            return nodes
        q_proj = getattr(getattr(module, "attn", None), "q_proj", None)
        if q_proj is None:
            return nodes

        filtered = []
        for node in nodes:
            layers = node.get("layers") or []
            if q_proj not in layers:
                filtered.append(node)
                continue
            remaining = [consumer for consumer in layers if consumer is not q_proj]
            if remaining:
                filtered.append({**node, "layers": remaining})
        return filtered

    def prepare_layer_replay_kwargs(
        self,
        layer,
        layer_input,
        additional_inputs,
        target_device,
    ):
        """Replay PanGu's "top query" layer with the input HF actually gives it.

        ``GPTPanguModel.forward`` applies ``ln_f`` before the last block and
        passes it ``custom_query = wqe(position_ids)``, so that block's
        ``q_proj`` reads query embeddings rather than the hidden states. Without
        this hook GPTQModel replayed layer 0's kwargs (``custom_query=None``)
        and chained the raw output, calibrating the last layer on the wrong
        input (last-layer arg0 rel diff 0.953, GPTQ error 0.114 vs RTN 0.097).
        """

        additional_inputs = super().prepare_layer_replay_kwargs(
            layer=layer,
            layer_input=layer_input,
            additional_inputs=additional_inputs,
            target_device=target_device,
        )

        transformer = getattr(self.model, "transformer", None)
        blocks = getattr(transformer, "h", None)
        if transformer is None or not blocks or blocks[-1] is not layer:
            return additional_inputs

        if layer_input:
            hidden_states = layer_input[0]
            norm = transformer.ln_f
            try:
                norm_device = next(norm.parameters()).device
            except StopIteration:
                norm_device = hidden_states.device
            if norm_device != hidden_states.device:
                norm = move_to(norm, device=hidden_states.device)
            layer_input[0] = norm(hidden_states)
            hidden_states = layer_input[0]
        else:
            hidden_states = None

        position_ids = additional_inputs.get("position_ids")
        if not torch.is_tensor(position_ids):
            if not torch.is_tensor(hidden_states):
                return additional_inputs
            seq_len = (
                int(hidden_states.shape[-2])
                if hidden_states.dim() >= 2
                else int(hidden_states.shape[0])
            )
            batch = int(hidden_states.shape[0]) if hidden_states.dim() >= 3 else 1
            position_ids = torch.arange(
                seq_len, device=hidden_states.device, dtype=torch.long
            ).unsqueeze(0).expand(batch, -1)

        additional_inputs["custom_query"] = transformer.wqe(
            position_ids.to(target_device)
        )
        return additional_inputs
