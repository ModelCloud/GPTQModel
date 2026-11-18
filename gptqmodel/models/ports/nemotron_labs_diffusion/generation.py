# SPDX-FileCopyrightText: 2026 NVIDIA CORPORATION & AFFILIATES
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Pinned NVIDIA block-diffusion, AR and self-speculative samplers; see NOTICE."""

from collections.abc import Callable, Iterator
from functools import wraps
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache

from .configuration_nemotron_labs_diffusion import NemotronLabsDiffusionConfig
from .modeling_ministral import Ministral3Model


def _restore_adapter_state(fn):
    @wraps(fn)
    def call(self, *args, **kwargs):
        states = [(m, m._disable_adapters) for m in self.modules() if hasattr(m, "_disable_adapters")]
        try:
            return fn(self, *args, **kwargs)
        finally:
            for module, state in states:
                module._disable_adapters = state
    return call


def _validate_decode(prompt_ids, max_new_tokens, temperature):
    if prompt_ids.ndim != 2 or prompt_ids.shape[1] == 0 or max_new_tokens < 1 or temperature < 0:
        raise ValueError("Decoding requires nonempty rank-two prompt, positive token budget and nonnegative temperature")


def _crop_dynamic_cache(cache, max_length):
    remove = cache.get_seq_length() - max_length
    if remove < 0:
        raise ValueError("Cannot grow a speculative cache by cropping")
    if remove:
        cache.crop(-remove)


class DiffusionGenerationMixin:
    config: NemotronLabsDiffusionConfig
    encoder: Ministral3Model
    diffusion_head: torch.nn.Module
    mask_token_id: int
    __call__: Callable[..., Any]
    if TYPE_CHECKING:
        def modules(self) -> Iterator[torch.nn.Module]: ...

    @torch.no_grad()
    @_restore_adapter_state
    def generate(
        self,
        prompt_ids: Optional[torch.Tensor] = None,
        max_new_tokens: int = 128,
        block_length: int = 32,
        threshold: Optional[float] = None,
        causal_context: bool = True,
        temperature: float = 0.0,
        eos_token_id: Optional[int] = None,
        max_thinking_tokens: Optional[int] = None,
        end_think_token_id: Optional[int] = None,
        *,
        input_ids: Optional[torch.Tensor] = None,
        inputs: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pad_token_id: Optional[int] = None,
    ):
        """Block-wise diffusion decoding with prefix-cached KV (LLaDA-style).

        Each block: append `block_length` mask tokens, then iteratively unmask
        by confidence top-k (with optional threshold). When `causal_context`,
        the KV cache and the next-block seed are produced via a causal forward
        between blocks (call-local `diffusion_lm`), matching the AR
        objective at block boundaries.

        Returns (output_ids, nfe) — output_ids includes the prompt.
        """
        candidates = [value for value in (prompt_ids, input_ids, inputs) if value is not None]
        if len(candidates) != 1:
            raise ValueError("Specify exactly one of prompt_ids, input_ids or inputs")
        prompt_ids = candidates[0]
        if attention_mask is not None:
            if attention_mask.shape != prompt_ids.shape or not attention_mask.bool().all():
                raise ValueError("Diffusion generation requires equal-length unpadded prompts")
        _validate_decode(prompt_ids, max_new_tokens, temperature)
        if block_length < 1 or max_new_tokens % block_length:
            raise ValueError("max_new_tokens must be divisible by positive block_length")
        if eos_token_id is None:
            eos_token_id = getattr(self.config, "eos_token_id", None)
        mask_id = self.mask_token_id

        x_accum = prompt_ids.clone()
        B = prompt_ids.shape[0]

        assert max_new_tokens % block_length == 0
        num_blocks = max_new_tokens // block_length
        # one denoising step per generated token (matches legacy chat_utils call)
        steps_per_block = block_length

        nfe = 0
        diffusion_lm = self.config.dlm_paradigm != "autoregressive"

        diffusion_lm = self.config.dlm_paradigm != "autoregressive"
        # Initial causal prefill produces the KV cache and the next-block seed.
        if causal_context:
            diffusion_lm = False
        output = self(prompt_ids, diffusion_lm=diffusion_lm, use_cache=True, use_causal_mask=causal_context)
        past_key_values = output.past_key_values
        if causal_context:
            diffusion_lm = True

        next_token = None
        if causal_context:
            last_logit = output.logits[:, -1, :]
            if temperature > 0:
                next_token = torch.multinomial(torch.softmax(last_logit / temperature, dim=-1), num_samples=1)
            else:
                next_token = torch.argmax(last_logit, dim=-1, keepdim=True)

        for num_block in range(num_blocks):
            mask_block = torch.full(
                (B, block_length), mask_id, dtype=prompt_ids.dtype, device=prompt_ids.device,
            )
            if causal_context:
                assert next_token is not None
                mask_block[:, 0] = next_token[:, 0]

            x_accum = torch.cat([x_accum, mask_block], dim=1)
            block_start = prompt_ids.size(1) + num_block * block_length
            block_slice = slice(block_start, block_start + block_length)

            # Thinking-budget enforcement: if we've passed max_thinking_tokens
            # without an end-think marker, inject one into this block.
            if end_think_token_id is not None and max_thinking_tokens is not None:
                tokens_before = num_block * block_length
                tokens_after = tokens_before + block_length
                if tokens_after > max_thinking_tokens:
                    gen_so_far = x_accum[:, prompt_ids.size(1):block_start]
                    has_end_think = (
                        (gen_so_far == end_think_token_id).any(dim=1)
                        if gen_so_far.size(1) > 0
                        else torch.zeros(B, dtype=torch.bool, device=prompt_ids.device)
                    )
                    if not has_end_think.all():
                        offset = max(0, max_thinking_tokens - tokens_before)
                        inject_pos = block_start + offset
                        for b in range(B):
                            if not has_end_think[b]:
                                x_accum[b, inject_pos] = end_think_token_id

            mask_block_idx0 = x_accum[:, block_slice] == mask_id
            num_transfer_tokens = _get_num_transfer_tokens(mask_block_idx0, steps_per_block)

            # Denoise the current block by repeated confidence-based unmasking.
            for i in range(steps_per_block):
                mask_block_idx = x_accum[:, block_slice] == mask_id
                if mask_block_idx.sum() == 0:
                    break

                nfe += 1
                logits_block = self(diffusion_lm=diffusion_lm,
                    input_ids=x_accum[:, block_slice],
                    past_key_values=past_key_values,
                    use_cache=False,
                ).logits

                x0, transfer_idx = _get_transfer_index(
                    logits_block, temperature, mask_block_idx, x_accum[:, block_slice],
                    num_transfer_tokens=num_transfer_tokens[:, i], threshold=threshold,
                )
                cur = x_accum[:, block_slice].clone()
                cur[transfer_idx] = x0[transfer_idx]
                x_accum[:, block_slice] = cur

                if eos_token_id is not None:
                    block_tokens = x_accum[:, block_slice]
                    eos_mask = block_tokens == eos_token_id
                    if eos_mask.any(dim=1).any():
                        after_eos = eos_mask.cumsum(dim=1).bool()
                        mask_before = (block_tokens == mask_id) & ~after_eos
                        if (eos_mask.any(dim=1) & ~mask_before.any(dim=1)).any():
                            break

            # Post-block: causal forward over the block to update the KV cache
            # and (when causal_context) sample the seed for the next block.
            if causal_context:
                diffusion_lm = False
            output = self(diffusion_lm=diffusion_lm,
                input_ids=x_accum[:, block_slice],
                past_key_values=past_key_values,
                use_cache=True,
                use_causal_mask=causal_context,
            )
            past_key_values = output.past_key_values
            nfe += 1

            if causal_context:
                diffusion_lm = True
                last_logit = output.logits[:, -1, :]
                if temperature > 0:
                    next_token = torch.multinomial(torch.softmax(last_logit / temperature, dim=-1), num_samples=1)
                else:
                    next_token = torch.argmax(last_logit, dim=-1, keepdim=True)

            if eos_token_id is not None:
                gen_so_far = x_accum[:, prompt_ids.size(1):]
                is_eos = gen_so_far == eos_token_id
                if is_eos.any(dim=1).all():
                    first_eos = is_eos.to(torch.int64).argmax(dim=1)
                    max_eos = first_eos.max().item()
                    return x_accum[:, : prompt_ids.size(1) + max_eos + 1], nfe

        return x_accum, nfe



    @torch.no_grad()
    @_restore_adapter_state
    def ar_generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        eos_token_id: Optional[int] = None,
        max_thinking_tokens: Optional[int] = None,
        end_think_token_id: Optional[int] = None,
    ) -> tuple:
        """Autoregressive generation calling the encoder directly (injected by build_hf_tidar_repo).

        Bypasses NemotronLabsDiffusionModel.forward() to avoid diffusion-specific
        code paths. Calls self.encoder (Ministral3Model) with explicit
        position_ids and use_cache so the KV cache and causal masking behave
        identically to MistralForCausalLM / vLLM.

        Returns:
            (output_ids, nfe) where output_ids includes the prompt.
        """
        diffusion_lm = False

        if eos_token_id is None:
            eos_token_id = getattr(self.config, 'eos_token_id', None)

        _validate_decode(prompt_ids, max_new_tokens, temperature)
        device = prompt_ids.device
        batch_size, prompt_len = prompt_ids.shape

        past_key_values = DynamicCache()
        position_ids = torch.arange(prompt_len, device=device).unsqueeze(0).expand(batch_size, -1)

        enc_out = self.encoder(diffusion_lm=diffusion_lm,
            input_ids=prompt_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = enc_out.past_key_values
        next_logit = self.diffusion_head(enc_out.last_hidden_state[:, -1:, :]).squeeze(1)

        generated_tokens = []
        nfe = 0

        for step in range(max_new_tokens):
            nfe += 1

            if temperature > 0:
                probs = torch.softmax(next_logit / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_logit, dim=-1, keepdim=True)

            # ---- thinking budget enforcement ----
            if end_think_token_id is not None and max_thinking_tokens is not None:
                if step >= max_thinking_tokens:
                    if generated_tokens:
                        gen_tensor = torch.cat(generated_tokens, dim=1)
                        has_end_think = (gen_tensor == end_think_token_id).any(dim=1)
                    else:
                        has_end_think = torch.zeros(batch_size, dtype=torch.bool, device=device)
                    for b in range(batch_size):
                        if not has_end_think[b]:
                            next_token[b] = end_think_token_id

            generated_tokens.append(next_token)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

            if step < max_new_tokens - 1:
                cur_pos = prompt_len + step
                step_pos_ids = torch.tensor([[cur_pos]], device=device).expand(batch_size, -1)

                enc_out = self.encoder(diffusion_lm=diffusion_lm,
                    input_ids=next_token,
                    position_ids=step_pos_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = enc_out.past_key_values
                next_logit = self.diffusion_head(enc_out.last_hidden_state[:, -1:, :]).squeeze(1)

        all_generated = torch.cat(generated_tokens, dim=1)
        output_ids = torch.cat([prompt_ids, all_generated], dim=1)
        return output_ids, nfe


    @torch.no_grad()
    @_restore_adapter_state
    def linear_spec_generate(
        self,
        prompt_ids: torch.Tensor,
        max_new_tokens: int = 128,
        block_length: int = 32,
        temperature: float = 0.0,
        mask_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        max_thinking_tokens: Optional[int] = None,
        end_think_token_id: Optional[int] = None,
        threshold: float = 0.0,
    ):
        """Linear speculative decoding: diffusion draft + AR verify.

        Each iteration: (1) draft the next block under bidirectional attention,
        (2) verify the drafted block under causal attention, accept the longest
        prefix where draft matches AR + one bonus token, advance the KV cache.

        LoRA-aware: when a PEFT adapter is attached to the model (e.g.
        ``linear_spec_lora``), it is toggled ON for the bidirectional draft
        phase and OFF for the causal prefill / verify phases — so the adapter
        only specializes the diffusion-mode forward and AR semantics are
        preserved. With no adapter loaded the calls are no-ops.

        Returns ``(output_ids, nfe)`` — ``output_ids`` includes the prompt.
        """
        if block_length < 2:
            raise ValueError("linear speculative block_length must be at least two")
        if prompt_ids.shape[0] != 1:
            raise ValueError("Linear speculative decoding requires batch_size == 1")

        token_mask_id = mask_token_id if mask_token_id is not None else self.config.mask_token_id
        if eos_token_id is None:
            eos_token_id = getattr(self.config, "eos_token_id", None)

        _validate_decode(prompt_ids, max_new_tokens, temperature)
        device = prompt_ids.device


        def _toggle_adapters(enable: bool):
            # No-op when no PEFT/LoRA modules are attached.
            for module in self.modules():
                if hasattr(module, "_disable_adapters"):
                    setattr(module, "_disable_adapters", not enable)

        # Prefill (causal, LoRA OFF).
        diffusion_lm = False
        _toggle_adapters(False)
        enc_out = self.encoder(diffusion_lm=diffusion_lm,
            input_ids=prompt_ids,
            past_key_values=DynamicCache(),
            use_cache=True,
            use_causal_mask=True,
        )
        past_key_values = enc_out.past_key_values
        last_logit = self.diffusion_head(enc_out.last_hidden_state[:, -1:, :]).squeeze(1)
        nfe = 1

        if temperature > 0:
            next_token = torch.multinomial(torch.softmax(last_logit / temperature, dim=-1), num_samples=1)
        else:
            next_token = torch.argmax(last_logit, dim=-1, keepdim=True)

        if eos_token_id is not None and next_token.item() == eos_token_id:
            return torch.cat([prompt_ids, next_token], dim=1), nfe

        generated = [next_token]
        total_gen = 1

        while total_gen < max_new_tokens:
            cache_len = past_key_values.get_seq_length()

            block = torch.full((1, block_length), token_mask_id, dtype=torch.long, device=device)
            block[0, 0] = next_token.item()
            is_mask = torch.ones_like(block, dtype=torch.bool)
            is_mask[:, 0] = False

            # Draft phase (bidirectional, LoRA ON) — iterate at threshold>0 so
            # that even low-confidence blocks make progress.
            diffusion_lm = True
            _toggle_adapters(True)
            while is_mask.any():
                enc_out = self.encoder(diffusion_lm=diffusion_lm, input_ids=block, past_key_values=past_key_values, use_cache=False)
                nfe += 1

                draft_logits = self.diffusion_head(enc_out.last_hidden_state)
                # LLaDA: logit[i] directly predicts position i — no shift needed.

                if temperature > 0:
                    draft_probs = torch.softmax(draft_logits / temperature, dim=-1)
                    draft_tokens = torch.multinomial(
                        draft_probs.view(-1, draft_probs.shape[-1]), num_samples=1
                    ).view(1, block_length)
                else:
                    draft_tokens = draft_logits.argmax(dim=-1)
                    draft_probs = torch.softmax(draft_logits, dim=-1)

                if threshold > 0:
                    draft_conf = torch.gather(draft_probs, -1, draft_tokens.unsqueeze(-1)).squeeze(-1)
                    draft_conf = torch.where(is_mask, draft_conf, -torch.inf)
                    unmask = draft_conf >= threshold
                    # Force progress even when every masked position is below threshold.
                    if not unmask.any():
                        best_idx = draft_conf.view(-1).argmax()
                        unmask = torch.zeros_like(is_mask, dtype=torch.bool)
                        unmask.view(-1)[best_idx] = True
                    block[unmask] = draft_tokens[unmask]
                    is_mask[unmask] = False
                else:
                    block[is_mask] = draft_tokens[is_mask]
                    break

            # Verify phase (causal, LoRA OFF).
            diffusion_lm = False
            _toggle_adapters(False)
            enc_out = self.encoder(diffusion_lm=diffusion_lm,
                input_ids=block,
                past_key_values=past_key_values,
                use_cache=True,
                use_causal_mask=True,
            )
            past_key_values = enc_out.past_key_values
            nfe += 1

            verify_logits = self.diffusion_head(enc_out.last_hidden_state)
            if temperature > 0:
                ar_tokens = torch.multinomial(
                    torch.softmax(verify_logits / temperature, dim=-1).view(-1, verify_logits.shape[-1]),
                    num_samples=1,
                ).view(1, block_length)
            else:
                ar_tokens = verify_logits.argmax(dim=-1)

            # Accept consecutive matches; AR also gives one bonus token at the tail.
            accepted = 0
            for i in range(block_length - 1):
                if ar_tokens[0, i].item() == block[0, i + 1].item():
                    accepted += 1
                else:
                    break
            accepted += 1

            accepted_toks = ar_tokens[:, :accepted]
            generated.append(accepted_toks)
            total_gen += accepted

            _crop_dynamic_cache(past_key_values, cache_len + accepted)
            next_token = ar_tokens[:, accepted - 1 : accepted]

            if eos_token_id is not None:
                eos_pos = (accepted_toks[0] == eos_token_id).nonzero(as_tuple=True)[0]
                if len(eos_pos) > 0:
                    first_eos = eos_pos[0].item()
                    generated[-1] = accepted_toks[:, : first_eos + 1]
                    total_gen = total_gen - accepted + first_eos + 1
                    break

            # Thinking-budget enforcement: force end-think as next seed if budget exhausted.
            if end_think_token_id is not None and max_thinking_tokens is not None:
                if total_gen > max_thinking_tokens:
                    all_gen = torch.cat(generated, dim=1)
                    if not (all_gen == end_think_token_id).any():
                        next_token = torch.tensor([[end_think_token_id]], device=device)

            if total_gen >= max_new_tokens:
                break

        all_generated = torch.cat(generated, dim=1)
        output_ids = torch.cat([prompt_ids, all_generated], dim=1)
        return output_ids, nfe



def _add_gumbel_noise(logits, temperature):
    """Gumbel-max sampling in float64 (low-precision Gumbel hurts MDM quality)."""
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def _get_num_transfer_tokens(mask_index, steps: int):
    """Even split of masked positions across `steps`, with remainder front-loaded."""
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : int(remainder[i])] += 1
    return num_transfer_tokens


def _get_transfer_index(logits, temperature, mask_index, x, num_transfer_tokens, threshold=None):
    """Pick which masked positions to commit this denoising step.

    Returns (x0, transfer_index): x0 is argmax tokens (clamped to original x at
    non-masked positions); transfer_index is a bool mask over positions to
    finalize, chosen by top-k confidence (and filtered by `threshold` if given).
    """
    logits_with_noise = _add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)

    p = F.softmax(logits, dim=-1)
    x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)

    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -torch.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True
        if threshold is not None:
            for k in range(1, num_transfer_tokens[j]):
                if confidence[j, select_index[k]] < threshold:
                    transfer_index[j, select_index[k]] = False
    return x0, transfer_index


def gumbel_topk(log_w: torch.Tensor, k: int) -> torch.Tensor:
    """Return a Bool mask of length len(log_w) with exactly k True."""
    g = -torch.log(-torch.log(torch.rand_like(log_w) + 1e-9) + 1e-9)
    topk = torch.topk(log_w + g, k).indices
    mask = torch.zeros_like(log_w, dtype=torch.bool)
    mask[topk] = True
    return mask
