"""Explicit full-prompt V4.1 layer state for causal calibration.

Each batch owns the mHC carry, compressed KV, index keys, candidate selections,
and its PLE gathers. No model-global reference may substitute for this state.
"""

from dataclasses import dataclass, field

import torch


def owned_tree(value, device):
    """Copy every tensor into owned storage; reject hidden mutable objects."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device, copy=True)
    if isinstance(value, dict):
        return {key: owned_tree(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(owned_tree(item, device) for item in value)
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise TypeError(f"unsupported V4.1 replay state: {type(value).__name__}")


@dataclass
class V41ReplayBatch:
    next_layer: int
    hidden: torch.Tensor
    pre_mix: torch.Tensor
    shared: dict
    kwargs: dict
    engram_rows: dict
    target_layer_ids: tuple = ()
    target_features: dict = field(default_factory=dict)

    @classmethod
    @torch.no_grad()
    def prepare(cls, model, input_ids, attention_mask=None):
        """Use the model's own input preparation, stopping before decoder work.

        Source loading must already have replaced both PLE embeddings with mapped
        modules. This function never constructs a model or loads its parameters.
        """
        text = model.model
        gathered = {}
        captured = {}

        class Captured(Exception):
            pass

        def capture_layer(layer, args, kwargs):
            hidden, pre_mix, _, token_mask = args
            if token_mask is not None or kwargs.get("past_key_values") is not None:
                raise ValueError("full-prompt text calibration requires no decode cache")
            captured.update(hidden=hidden, pre_mix=pre_mix, kwargs=kwargs)
            raise Captured()

        handles = []
        try:
            for name, table in text.engram_tables.items():
                def capture_rows(module, args, output, layer=int(name)):
                    gathered[layer] = output
                handles.append(table.register_forward_hook(capture_rows))
            handles.append(text.layers[0].register_forward_pre_hook(capture_layer, with_kwargs=True))
            try:
                model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            except Captured:
                pass
        finally:
            for handle in handles:
                handle.remove()
        if not captured:
            raise RuntimeError("first V4.1 decoder layer was not reached")
        kwargs = dict(captured["kwargs"])
        shared = kwargs.pop("shared")
        return cls(0, owned_tree(captured["hidden"], "cpu"),
                   owned_tree(captured["pre_mix"], "cpu"), owned_tree(shared, "cpu"),
                   owned_tree(kwargs, "cpu"), owned_tree(gathered, "cpu"),
                   tuple(text.config.dspark_target_layer_ids))

    @torch.no_grad()
    def advance(self, layer, device):
        """Run one block from an immutable input; return owned CPU output state.

        Repeated candidate/replay evaluations start from the same shared-state
        snapshot, so one evaluation cannot change the next one's attention.
        """
        if layer.layer_idx != self.next_layer:
            raise ValueError(f"expected V4.1 layer {self.next_layer}, got {layer.layer_idx}")
        shared = owned_tree(self.shared, device)
        kwargs = owned_tree(self.kwargs, device)
        rows = self.engram_rows.get(self.next_layer)
        targets = dict(self.target_features)
        handle = None
        if self.next_layer in self.target_layer_ids:
            # This is after any Engram injection, before the block's attention:
            # the reference uses the unweighted BF16 stream mean, not hc_collapse.
            def capture(module, args):
                targets[self.next_layer] = owned_tree(args[0].mean(dim=2), "cpu")
            handle = layer.attn_hc.register_forward_pre_hook(capture)
        try:
            hidden, pre_mix = layer(
                self.hidden.to(device), self.pre_mix.to(device),
                None if rows is None else rows.to(device), None,
                shared=shared, **kwargs,
            )
        finally:
            if handle is not None:
                handle.remove()
        # Consumed PLE rows must not survive in the next durable frontier.
        remaining = {index: value for index, value in self.engram_rows.items()
                     if index > self.next_layer}
        return type(self)(self.next_layer + 1, owned_tree(hidden, "cpu"),
                          owned_tree(pre_mix, "cpu"), owned_tree(shared, "cpu"),
                          self.kwargs, remaining, self.target_layer_ids, targets)
