"""Explicit dSpark calibration input preparation, without tokenizer policy."""

import torch
from torch import nn

from .v41_replay import V41ReplayBatch, owned_tree


class V41DSparkInput(nn.Module):
    def __init__(self, embedding, main_proj, main_norm, config):
        super().__init__()
        self.embed = embedding
        self.main_proj = main_proj
        self.main_norm = main_norm
        self.target_layer_ids = tuple(config.dspark_target_layer_ids)
        self.block_size = config.dspark_block_size
        self.noise_token_id = config.dspark_noise_token_id
        self.hc_mult = config.hc_mult

    @torch.no_grad()
    def forward(self, main_hidden, input_ids):
        """Project main features and embed the known token followed by noise.

        input_ids is one already-tokenized known token per batch item. main_hidden
        holds stream means at target-layer inputs in checkpoint-declared order.
        It may include the full main history needed to reconstruct attention.
        """
        if input_ids.ndim != 1 or input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("dSpark requires one integer token ID per batch item")
        if main_hidden.ndim != 3 or main_hidden.shape[0] != input_ids.shape[0]:
            raise ValueError("dSpark main features must have shape [batch, history, features]")
        main_x = self.main_norm(self.main_proj(main_hidden))
        draft_ids = input_ids.new_full((input_ids.shape[0], self.block_size), self.noise_token_id)
        draft_ids[:, 0] = input_ids
        hidden = self.embed(draft_ids).unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        return hidden, main_x

    @torch.no_grad()
    def prepare_joint(self, target_features, token_ids, *, positions):
        """One original prompt, all selected anchors, one projected main prefix."""
        if (token_ids.ndim != 2 or token_ids.shape[0] != 1
                or token_ids.dtype not in (torch.int32, torch.int64)):
            raise ValueError("joint dSpark preparation requires one integer-token source prompt")
        if (positions.ndim != 1 or positions.dtype != torch.long or not positions.numel()
                or positions.min() < 1 or positions.max() + 1 >= token_ids.shape[1]
                or not torch.all(positions[1:] > positions[:-1])):
            raise ValueError("joint anchors must be sorted distinct eligible positions")
        length = int(positions[-1]) + 1
        features = []
        for layer in self.target_layer_ids:
            value = target_features[layer]
            if value.ndim != 3 or value.shape[0] != 1 or value.shape[1] < length:
                raise ValueError("joint dSpark input lacks the shared target prefix")
            features.append(value[:, :length])
        device = self.main_norm.weight.device
        main = torch.cat([value.to(device) for value in features], dim=-1)
        main_x = self.main_norm(self.main_proj(main))
        draft_ids = torch.full((positions.numel(), self.block_size), self.noise_token_id,
                               dtype=torch.long, device=device)
        draft_ids[:, 0] = token_ids[0, positions.to(token_ids.device) + 1].to(device)
        hidden = self.embed(draft_ids).unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        pre = torch.zeros(*hidden.shape[:3], device=device, dtype=torch.float32)
        pre[..., 0] = 1
        return V41ReplayBatch(0, owned_tree(hidden, "cpu"), owned_tree(pre, "cpu"), {},
                              {"main_x": owned_tree(main_x, "cpu"),
                               "anchor_positions": owned_tree(positions, "cpu")}, {})

    @torch.no_grad()
    def prepare(self, target_features, token_ids, *, position):
        """Build owned replay for one teacher-forced draft position.

        Positions 0..position are visible main history. The known draft token
        is token_ids[:, position + 1], substituting the corpus token for the main
        model's sampled next token. Future main features are sliced away before
        projection; later draft tokens are noise, not teacher-forced answers.
        """
        if type(position) is not int or position < 1:
            raise ValueError("dSpark routed calibration requires position >= 1")
        if token_ids.ndim != 2 or token_ids.shape[1] <= position + 1:
            raise ValueError("dSpark requires corpus tokens including the next known token")
        features = []
        for layer in self.target_layer_ids:
            value = target_features[layer]
            if value.ndim != 3 or value.shape[1] <= position:
                raise ValueError("missing dSpark target history")
            features.append(value[:, :position + 1])
        device = self.main_norm.weight.device
        main_hidden = torch.cat([value.to(device) for value in features], dim=-1)
        hidden, main_x = self(main_hidden, token_ids[:, position + 1].to(device))
        pre = torch.zeros(*hidden.shape[:3], device=device, dtype=torch.float32)
        pre[..., 0] = 1
        return V41ReplayBatch(0, owned_tree(hidden, "cpu"), owned_tree(pre, "cpu"), {},
                              {"main_x": owned_tree(main_x, "cpu")}, {})
