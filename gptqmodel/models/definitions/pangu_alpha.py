# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import torch

from ...utils.model import move_to
from ..base import BaseQModel
from ..local import LocalModelLoader


class PanguAlphaQModel(BaseQModel):
    loader = LocalModelLoader(
        config_class="gptqmodel.models.ports.pangu_alpha.GPTPanguConfig",
        model_class="gptqmodel.models.ports.pangu_alpha.GPTPanguForCausalLM",
        tokenizer_class="gptqmodel.models.ports.pangu_alpha.tokenization_pangu.GPTPanguTokenizer",
    )
    require_trust_remote_code = False
    fusable = None
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
            "attn": ("attn:!", "q_proj:0", "k_proj:0", "v_proj:0", "c_proj:1"),
            "ln_2": ("ln_2:!",),
            # `GPTPanguMLP.forward` is `c_proj(act(c_fc(x)))` with GELU, which is
            # not positively homogeneous: folding a scale into `c_fc` changes the
            # MLP output (4.5e-2 on a tiny model).
            "mlp": ("c_fc:0:nosrc", "c_proj:1"),
        },
    ]

    def awq_get_modules_for_scaling(self, module, input_feat, module_kwargs):
        """Plain AWQ only: top-query replay/fold search has not been qualified."""
        return []

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

        # In a one-block model the captured block is already the top-query block.
        if additional_inputs.get("custom_query") is not None:
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
            seq_len = int(hidden_states.shape[-2]) if hidden_states.dim() >= 2 else int(hidden_states.shape[0])
            batch = int(hidden_states.shape[0]) if hidden_states.dim() >= 3 else 1
            position_ids = (
                torch.arange(seq_len, device=hidden_states.device, dtype=torch.long).unsqueeze(0).expand(batch, -1)
            )

        query = transformer.wqe(position_ids.to(target_device))
        if torch.is_tensor(hidden_states) and hidden_states.ndim == 3:
            query = query.expand(hidden_states.shape[0], -1, -1)
        additional_inputs["custom_query"] = query
        return additional_inputs
