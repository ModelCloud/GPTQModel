# SPDX-FileCopyrightText: 2026 Microsoft
# SPDX-License-Identifier: Apache-2.0
# Adapted from microsoft/Mage-VL@d88b153285f1633a61b2f693c59c8576693af185.
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..native.qwen3.configuration_qwen3 import Qwen3Config
from ..native.qwen3.modeling_qwen3 import Qwen3ForCausalLM


class PreNet(nn.Module):
    def __init__(self, d_code, d_model):
        super().__init__()
        self.fc3 = nn.Linear(d_code, d_model)

    def forward(self, x):
        return F.leaky_relu(self.fc3(x))


class PostNet(nn.Module):
    def __init__(self, d_model, n_class):
        super().__init__()
        self.fc3 = nn.Linear(d_model, n_class)

    def forward(self, x):
        return self.fc3(F.leaky_relu(x))


@dataclass
class SSMConfig:
    d_model: int = 2560
    n_ssm: int = 1


class VideoMamba(nn.Module):
    def __init__(self, config):
        super().__init__()
        try:
            from mamba_ssm.models.mixer_seq_simple import create_block  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError("Mage-VL StreamMind requires the installed mamba-ssm package and its device kernels.") from exc
        self.ssms = nn.ModuleList(
            [create_block(config.d_model, d_intermediate=0, layer_idx=i) for i in range(config.n_ssm)]
        )
        self.norm_fn = nn.LayerNorm(config.d_model)

    def forward(self, embeds, inference_params=None):
        hidden_states = embeds
        residual = None
        for ssm in self.ssms:
            hidden_states, residual = ssm(
                hidden_states, residual, inference_params=inference_params
            )
        residual = hidden_states + residual if residual is not None else hidden_states
        return self.norm_fn(residual.to(dtype=self.norm_fn.weight.dtype))


class Qwen3ForCausalLMCls(Qwen3ForCausalLM):
    # Deliberate classifier interface; preserve the checkpoint's native Qwen3 owners.
    def forward(self, inputs_embeds=None, labels=None, attention_mask=None, **kwargs):  # pyright: ignore[reportIncompatibleMethodOverride]
        outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        logits = self.lm_head(outputs.last_hidden_state).float()
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous().view(-1, self.config.vocab_size)
            shift_labels = labels[..., 1:].contiguous().view(-1).to(shift_logits.device)
            loss = nn.CrossEntropyLoss(
                weight=torch.tensor([0.15, 0.85], device=shift_logits.device)
            )(shift_logits, shift_labels)
        return {"loss": loss, "logits": logits}


class ClsNet(nn.Module):
    def __init__(self, hidden_size=2560, num_layers=4):
        super().__init__()
        config = Qwen3Config(
            vocab_size=2,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=32,
            num_key_value_heads=8,
            intermediate_size=12288,
            head_dim=128,
            max_position_embeddings=8192,
            rms_norm_eps=1e-6,
            tie_word_embeddings=False,
            attention_bias=False,
        )
        self.cls_model = Qwen3ForCausalLMCls(config)

    def forward(self, x, labels=None, attention_mask=None):
        return self.cls_model(inputs_embeds=x, labels=labels, attention_mask=attention_mask)


class StreamMindGate(nn.Module):
    def __init__(self, hidden_size=2560):
        super().__init__()
        self.pre_net = PreNet(hidden_size, hidden_size)
        self.mamba_model = VideoMamba(SSMConfig(d_model=hidden_size))
        self.post_net = PostNet(hidden_size, hidden_size)
        self.cls_net = ClsNet(hidden_size=hidden_size, num_layers=4)

    def perception_tokens(self, vision_tokens):
        """Convert [B,T,P,D] visual patches to one EPFE token per time step."""
        x = vision_tokens.mean(dim=2)
        batch, time, dim = x.shape
        x = self.pre_net(x.reshape(batch * time, dim)).reshape(batch, time, dim)
        x = self.mamba_model(x)
        x = self.post_net(x.reshape(batch * time, dim)).reshape(batch, time, dim)
        return x

    def forward(self, vision_tokens, response_positions=None):
        """Return [B,T,2] silent/speak logits for every EPFE time step."""
        tokens = self.perception_tokens(vision_tokens)
        batch, time, dim = tokens.shape
        target_ids = torch.zeros(batch, time, dtype=torch.long, device=tokens.device)
        if response_positions is not None:
            target_ids[:, torch.as_tensor(response_positions, device=tokens.device) - 1] = 1
        targets = self.cls_net.cls_model.model.embed_tokens(
            target_ids.reshape(batch * time)
        )
        pair = torch.stack((tokens.reshape(batch * time, dim), targets), dim=1)
        rotary = self.cls_net.cls_model.model.rotary_emb
        saved_inv_freq = rotary.inv_freq
        try:
            # Match the training checkpoint, where the full model (including
            # non-persistent Qwen3 RoPE buffers) was cast to BF16.
            rotary.register_buffer("inv_freq", rotary.inv_freq.to(pair.dtype), persistent=False)
            output = self.cls_net(
                pair,
                attention_mask=torch.ones(pair.shape[:2], device=pair.device),
            )
        finally:
            rotary.register_buffer("inv_freq", saved_inv_freq, persistent=False)
        # Autoregressive shift: position 0 predicts the target token at
        # position 1, matching StreamMind's logits[..., :-1, :] evaluation.
        return output["logits"][:, 0].reshape(batch, time, 2)
