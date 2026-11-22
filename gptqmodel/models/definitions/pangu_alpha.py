# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import torch

from ...utils.model import has_cpu_offload_hooks, move_to
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
            # not positively homogeneous. Using `c_fc` as a fold *source*
            # (`c_fc -> c_proj`) changes the MLP output (1.2e-2 on a tiny
            # model), so it is `nosrc`. Folding `ln_2`'s output *into* `c_fc`
            # is neutral and stays in the generic planner.
            "mlp": ("c_fc:0:nosrc", "c_proj:1"),
        },
    ]

    def awq_get_modules_for_scaling(self, module, input_feat, module_kwargs):
        """Drop the top-query layer's `q_proj` from the `ln_1` scaling group.

        ``GPTPanguModel.forward`` applies ``ln_f`` before the last block and
        passes it ``custom_query = wqe(position_ids)``, so that block's
        ``q_proj`` reads query embeddings rather than ``ln_1``'s output.
        Scaling it with the ``ln_1`` group changes the attention output
        (q is multiplied by the scale while its input is not divided), so it
        is removed from the node; the remaining k/v consumers keep the group.

        Every group the generic planner emits for PanGu is neutral on a real
        forward (``ln_1 -> q/k/v``, ``attn.v_proj -> attn.c_proj`` and
        ``ln_2 -> mlp.c_fc``); the non-neutral ``c_fc -> c_proj`` fold is
        already suppressed by the tree's ``nosrc`` flag. This is validated by
        ``tests/test_pangu_alpha_local_port.py`` for the first and last block.
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

        # In a one-block model the captured block is already the top-query block.
        if additional_inputs.get("custom_query") is not None:
            return additional_inputs

        if layer_input:
            hidden_states = layer_input[0]
            norm = self._materialize_replay_module(
                transformer.ln_f, device=hidden_states.device
            )
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

        replay_device = target_device
        if replay_device is None:
            replay_device = hidden_states.device if torch.is_tensor(hidden_states) else position_ids.device
        # ``wqe`` is not part of the quantized layer tree: lazy loading leaves it
        # on meta and disk offload leaves a meta placeholder behind an
        # Accelerate hook. Place it next to the replayed query before the
        # embedding lookup so the index and weight devices agree (#5135).
        wqe = self._materialize_replay_module(transformer.wqe, device=replay_device)
        query = wqe(position_ids.to(replay_device))
        if torch.is_tensor(hidden_states) and hidden_states.ndim == 3:
            query = query.expand(hidden_states.shape[0], -1, -1)
        additional_inputs["custom_query"] = query
        return additional_inputs

    def _materialize_replay_module(self, module, device):
        """Put a non-layer replay module on ``device`` from its loaded source.

        ``ln_f`` and ``wqe`` live outside the decoder-layer module tree, so the
        quantizer never materializes them. With lazy loading they stay meta
        shells; with ``offload_to_disk`` they are meta placeholders behind an
        Accelerate hook (possibly wrapped in ``SequentialHook``). ``move_to``
        alone cannot handle either: it refuses a meta module whose hook it does
        not recognize, and a lazy shell has no hook to restore from. Use the
        same helpers the looper uses, then relocate the materialized module.
        """
        from ...utils.offload import is_meta_module, undo_offload_to_disk

        if is_meta_module(module):
            if has_cpu_offload_hooks(module):
                module = undo_offload_to_disk(module, device=device)
            else:
                materialize = getattr(self, "shell_module_materialize", None)
                if getattr(self, "turtle_model", None) is not None and callable(
                    materialize
                ):
                    module = materialize(module, device=device)
        return move_to(module, device=device)
