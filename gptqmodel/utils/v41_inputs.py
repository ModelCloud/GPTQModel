"""Bounded native V4.1 input preparation, without constructing decoder blocks."""

import torch
from torch import nn

from .v41_replay import V41ReplayBatch, owned_tree


class V41MainInput(nn.Module):
    """Own only token embeddings, small hash/rotary state and mapped PLE handles.

    This adapter consumes unpadded, independent, equal-length text sequences.
    It never packs examples together and supports joint batches. Native attention
    constructs sliding indices directly, so no quadratic causal mask is stored.
    """

    def __init__(self, embedding, hash_state, rotary, tables, config):
        super().__init__()
        self.embed = embedding
        self.hash_state = hash_state
        self.rotary = rotary
        self.tables = nn.ModuleDict(tables)
        self.config = config

    @torch.no_grad()
    def prepare(self, input_ids):
        if input_ids.ndim != 2 or not all(input_ids.shape) or input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("main replay requires nonempty [batch, sequence] integer token IDs")
        device = self.embed.weight.device
        input_ids = input_ids.to(device)
        embedded = self.embed(input_ids)
        hidden = embedded.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()
        pre = torch.zeros(*hidden.shape[:-1], device=device, dtype=torch.float32)
        pre[..., 0] = 1
        positions = torch.arange(input_ids.shape[1], device=device)[None].expand(input_ids.shape[0], -1)
        rotary = {kind: self.rotary(embedded, position_ids=positions, layer_type=kind)
                  for kind in ("main", "compress")}
        rows = {}
        if self.hash_state is not None:
            hashes = self.hash_state(input_ids, None, None)
            for index, layer in enumerate(self.config.engram_layer_ids):
                # Store owned CPU rows immediately; never retain both lookups on GPU.
                rows[layer] = owned_tree(self.tables[str(layer)](hashes[:, :, index]), "cpu")
        kwargs = {"position_embeddings": owned_tree(rotary, "cpu"),
                      "position_ids": owned_tree(positions, "cpu"), "attention_mask": None,
                      "padding_mask": None, "past_key_values": None}
        return V41ReplayBatch(0, owned_tree(hidden, "cpu"), owned_tree(pre, "cpu"), {}, kwargs,
                              rows, tuple(self.config.dspark_target_layer_ids))

    def close(self):
        for table in self.tables.values():
            table.close()
