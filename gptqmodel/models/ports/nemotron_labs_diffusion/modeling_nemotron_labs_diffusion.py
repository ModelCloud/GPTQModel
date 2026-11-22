# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Package-owned NVIDIA tri-mode diffusion model, adapted to Transformers 5."""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutput
from transformers.utils import can_return_tuple
from transformers.utils.generic import ModelOutput

from .configuration_nemotron_labs_diffusion import NemotronLabsDiffusionConfig
from .generation import DiffusionGenerationMixin
from .modeling_ministral import Ministral3Model, Ministral3PreTrainedModel


@dataclass
class NemotronLabsDiffusionOutputWithPast(ModelOutput):
    loss: object = None
    logits: torch.Tensor | None = None
    causal_logits: torch.Tensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None
    attentions: tuple[torch.Tensor, ...] | None = None


class NemotronLabsDiffusionModel(Ministral3PreTrainedModel, DiffusionGenerationMixin):
    _tied_weights_keys = {"diffusion_head.weight": "encoder.embed_tokens.weight"}

    def __init__(self, config: NemotronLabsDiffusionConfig):
        super().__init__(config)
        self.mask_token_id = config.mask_token_id
        self.encoder = Ministral3Model(config)
        self.diffusion_head: nn.Module = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.vocab_size = config.vocab_size
        self.post_init()

    def get_input_embeddings(self):
        return self.encoder.embed_tokens

    def set_input_embeddings(self, value):
        self.encoder.embed_tokens = value

    def get_output_embeddings(self):
        return self.diffusion_head

    def set_output_embeddings(self, new_embeddings):
        self.diffusion_head = new_embeddings

    def forward_process(self, input_ids, eps=1e-3, block_size=None, loss_mask=None):
        b, l = input_ids.shape
        device = input_ids.device

        if self.config.dp_varying_mask_ratio:
            # Enable different random seeds for each DP rank during sampling
            import torch.distributed as dist
            dp_rank = 0
            if dist.is_initialized():
                try:
                    dp_rank = dist.get_rank()
                except Exception:
                    dp_rank = 0
            # Use a local generator to avoid affecting global RNG state
            generator = torch.Generator(device=device)
            generator.manual_seed(torch.seed() + dp_rank)
        else:
            generator = None

        t = torch.rand(b, device=device, generator=generator)

        p_mask = (1 - eps) * t + eps  # shape: (b,)
        p_mask = p_mask[:, None].expand(-1, l)  # shape: (b, l)

        masked_indices = torch.rand((b, l), device=device) < p_mask

        if loss_mask is not None:
            masked_indices[loss_mask == 0] = 0

        noisy_batch = torch.where(masked_indices, self.mask_token_id, input_ids)

        return noisy_batch, masked_indices, p_mask


    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor]   = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor]       = None,
        split_len: Optional[int]                 = None,
        past_key_values: Optional[Cache]         = None,
        block_size: Optional[int]                = None,
        eps: float                               = 1e-3,
        is_teacher: bool                        = False,
        masked_indices: Optional[torch.Tensor]   = None,
        p_mask: Optional[torch.Tensor]           = None,
        teacher_logits: Optional[torch.Tensor]   = None,
        masked_indices_teacher: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        ce_loss_weight: float = 1.0,
        output_last_hidden_states_only: bool = False,
        skip_loss: bool = False,
        **kwargs,
    ) -> NemotronLabsDiffusionOutputWithPast | BaseModelOutput:

        batch_size, seq_len = input_ids.shape

        if self.config.dlm_paradigm == 'block_diff':
            if labels is not None and block_size is None:
                block_size = self.config.block_size
        elif self.config.dlm_paradigm not in ('bidirectional', 'autoregressive'):
            raise ValueError(f"Unknown dLM paradigm: {self.config.dlm_paradigm}")

        if labels is not None and self.config.dlm_paradigm != 'autoregressive':
            if masked_indices is not None:
                if p_mask is None and not skip_loss:
                    raise ValueError("p_mask is required with explicit masked_indices")
                masked_indices = masked_indices.clone()

                if loss_mask is not None:
                    masked_indices[loss_mask == 0] = 0

                noisy_inputs = torch.where(masked_indices, self.mask_token_id, input_ids)

            else:
                noisy_inputs, masked_indices, p_mask = self.forward_process(input_ids, eps=eps, block_size=block_size, loss_mask=loss_mask)

        else:
            noisy_inputs = input_ids
            masked_indices = None
            p_mask = None

        input_ids_len = noisy_inputs.shape[1]
        if labels is not None and self.config.dlm_paradigm == 'block_diff':
            if position_ids is None:
                position_ids = torch.arange(input_ids_len, device=noisy_inputs.device).unsqueeze(0)
            noisy_inputs = torch.cat([noisy_inputs, input_ids], dim=1)

        enc_out  = self.encoder(
            past_key_values=past_key_values,
            input_ids=noisy_inputs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            is_training=(labels is not None),
            return_dict=True,
            **kwargs,
        )

        if output_last_hidden_states_only:
            return BaseModelOutput(last_hidden_state=enc_out.last_hidden_state)

        logits = self.diffusion_head(enc_out.last_hidden_state)  # (batch, len_B, vocab)
        causal_logits = None

        if labels is not None and self.config.dlm_paradigm == 'block_diff':
            causal_logits = logits[:, input_ids_len:]
            logits = logits[:, :input_ids_len]

        loss = None
        if labels is not None and not skip_loss:
            if self.config.dlm_paradigm == 'autoregressive':
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                if loss_mask is None:
                    loss_fct = CrossEntropyLoss()
                    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                    shift_labels = shift_labels.view(-1)
                    loss = loss_fct(shift_logits, shift_labels)

                else:
                    loss_mask = loss_mask[..., 1:].contiguous()

                    loss_fct = CrossEntropyLoss(reduction='none')
                    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                    shift_labels = shift_labels.view(-1)
                    shift_labels = shift_labels.to(shift_logits.device)

                    token_losses = loss_fct(shift_logits, shift_labels)

                    flat_loss_mask = loss_mask.reshape(-1)
                    loss = token_losses[flat_loss_mask == 1].sum() / flat_loss_mask.sum()

            else:
                assert masked_indices is not None and p_mask is not None
                # LLaDA-style diffusion loss on masked positions.
                # Token-wise cross entropy loss on masked positions.
                token_loss = torch.nn.functional.cross_entropy(
                    logits[masked_indices],
                    labels[masked_indices],
                    reduction='none'
                ) / p_mask[masked_indices]

                num_mask_tokens = masked_indices.sum()

                # global_loss_avg=True: loss is reduced externally by global token count.
                loss = token_loss.sum()

                if self.config.dlm_loss_weight is not None:
                    loss = self.config.dlm_loss_weight * loss

                if self.config.dlm_paradigm == 'block_diff':
                    assert causal_logits is not None
                    # AR-side loss for block-diffusion paradigm.
                    causal_logits = causal_logits[..., :-1, :].contiguous()
                    causal_logits = causal_logits.view(-1, causal_logits.size(-1))
                    causal_labels = labels[..., 1:].contiguous().view(-1)

                    loss_fct = CrossEntropyLoss(reduction='sum')
                    ar_loss = loss_fct(causal_logits, causal_labels)

                    self.loss_diffusion = loss.detach().item() / num_mask_tokens
                    self.loss_ar = ar_loss.detach().item() / seq_len

                    loss = loss + self.config.ar_loss_weight * ar_loss

                # global_loss_avg=True: return (sum_loss, token_count) for external mean.
                if self.config.dlm_paradigm == 'block_diff':
                    loss = (loss, num_mask_tokens + int(self.config.ar_loss_weight * seq_len))
                else:
                    loss = (loss, num_mask_tokens)

        return NemotronLabsDiffusionOutputWithPast(
            loss=loss if not is_teacher else logits,
            logits=logits,
            causal_logits=causal_logits,
            past_key_values=enc_out.past_key_values,
            hidden_states=enc_out.hidden_states,
            attentions=enc_out.attentions,
        )
