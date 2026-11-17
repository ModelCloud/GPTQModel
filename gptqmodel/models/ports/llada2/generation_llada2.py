# Copyright 2025 Antgroup and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
# Adapted for Transformers 5 by ModelCloud.ai, 2026. See README.md and NOTICE.
"""Pinned InclusionAI diffusion/edit decoders. Version selection belongs to config."""

from typing import Optional

import torch
import torch.nn.functional as F
from transformers.utils import logging


logger = logging.get_logger(__name__)


class LLaDA20Generation:
    @staticmethod
    def _top_k_logits(logits, k):
        if k is None or k <= 0:
            return logits
        else:
            values, _ = torch.topk(logits, k)
            min_values = values[..., -1, None]
            return torch.where(logits < min_values, torch.full_like(logits, float("-inf")), logits)

    @staticmethod
    def _top_p_logits(logits, p):
        if p is None or p >= 1.0:
            return logits
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_mask = cumulative_probs > p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        mask_indices = torch.scatter(torch.full_like(logits, False, dtype=torch.bool), -1, sorted_indices, sorted_mask)
        return logits.masked_fill(mask_indices, float("-inf"))

    def _sample_with_temperature_topk_topp(self, logits, temperature=1.0, top_k=0, top_p=1.0):
        orig_shape = logits.shape[:-1]
        vocab_size = logits.shape[-1]
        logits = logits.reshape(-1, vocab_size)
        if temperature > 0 and temperature != 1.0:
            logits = logits / temperature
        logits = self._top_k_logits(logits, top_k)
        logits = self._top_p_logits(logits, top_p)
        probs = F.softmax(logits, dim=-1)
        token = torch.multinomial(probs, num_samples=1)
        token_prob = torch.gather(probs, -1, token)
        return (token.view(*orig_shape), token_prob.view(*orig_shape))

    @staticmethod
    def _get_num_transfer_tokens(block_length, steps):
        if steps == 0:
            return torch.tensor([], dtype=torch.int64)
        base = block_length // steps
        remainder = block_length % steps
        num_transfer_tokens = torch.full((steps,), base, dtype=torch.int64)
        num_transfer_tokens[:remainder] += 1
        return num_transfer_tokens

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        temperature: int = 0.0,
        block_length: int = 32,
        steps: int = 32,
        gen_length: int = 2048,
        top_p: Optional[int] = None,
        top_k: Optional[int] = None,
        eos_early_stop: bool = False,
        minimal_topk: int = 1,
        threshold: float = 0.95,
        eos_id: int = 156892,
        mask_id: int = 156895,
    ):
        steps = min(steps, gen_length // minimal_topk)
        input_ids = inputs.to(self.device)
        prompt_length = input_ids.shape[1]
        num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
        total_length = num_blocks * block_length
        block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=self.device))
        block_diffusion_attention_mask = (
            block_mask.repeat_interleave(block_length, dim=0)
            .repeat_interleave(block_length, dim=1)
            .unsqueeze(0)
            .unsqueeze(0)
            .log()
            .to(torch.bfloat16)
        )
        position_ids = torch.arange(total_length, device=self.device).unsqueeze(0)
        x = torch.full((1, total_length), mask_id, dtype=torch.long, device=self.device)
        x[:, :prompt_length] = input_ids.clone()
        prompt_index_full = torch.zeros_like(x, dtype=torch.bool)
        prompt_index_full[:, :prompt_length] = True
        prefill_blocks = prompt_length // block_length
        denoising_steps_per_block = steps
        num_transfer_tokens_schedule = self._get_num_transfer_tokens(block_length, denoising_steps_per_block)
        for num_block in range(prefill_blocks, num_blocks):
            current_window_end = (num_block + 1) * block_length
            cur_x = x[:, :current_window_end]
            cur_attn_mask = block_diffusion_attention_mask[:, :, :current_window_end, :current_window_end]
            cur_position_ids = position_ids[:, :current_window_end]
            for step in range(denoising_steps_per_block):
                active_block_mask = cur_x[:, -block_length:] == mask_id
                if active_block_mask.sum() == 0:
                    break
                logits = self.forward(
                    cur_x, attention_mask=cur_attn_mask, position_ids=cur_position_ids, use_cache=False
                ).logits
                active_logits = logits[:, -block_length:, :]
                x0, x0_p = self._sample_with_temperature_topk_topp(
                    active_logits, temperature=temperature, top_k=top_k, top_p=top_p
                )
                num_to_transfer = num_transfer_tokens_schedule[step].item()
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                confidence = torch.where(active_block_mask, x0_p, -torch.inf)
                high_conf_mask = confidence[0] > threshold
                num_high_confidence = high_conf_mask.sum().item()
                if num_high_confidence >= num_to_transfer:
                    transfer_index[0] = high_conf_mask
                else:
                    _, idx = torch.topk(confidence[0], k=min(num_to_transfer, active_block_mask.sum().item()))
                    transfer_index[0, idx] = True
                if transfer_index.any():
                    cur_x[:, -block_length:][transfer_index] = x0[transfer_index]
                if eos_early_stop and (x0[transfer_index] == eos_id).any():
                    eos_pos_in_x = (cur_x[0] == eos_id).nonzero(as_tuple=True)
                    if len(eos_pos_in_x[0]) > 0:
                        eos_pos = eos_pos_in_x[0][0].item()
                        if (cur_x[0, prompt_length:eos_pos] != mask_id).all():
                            final_x = x[:, :total_length][:, : eos_pos + 1]
                            return final_x
            x[:, :current_window_end] = cur_x
            if eos_id is not None and (x[0, prompt_length:current_window_end] == eos_id).any():
                break
        generated_answer = x[:, : prompt_length + gen_length]
        mask_positions = (
            (generated_answer[0][input_ids.shape[1] :] == eos_id).nonzero(as_tuple=True)[0]
            if eos_id is not None
            else torch.empty(0, dtype=torch.long)
        )
        if len(mask_positions) > 0:
            first_mask_position = mask_positions[0].item()
        else:
            first_mask_position = gen_length
        return generated_answer[:, input_ids.shape[1] : input_ids.shape[1] + first_mask_position + 1]


class LLaDA20PreviewGeneration(LLaDA20Generation):
    def _sample_with_temperature_topk_topp(self, logits, temperature=1.0, top_k=0, top_p=1.0):
        orig_shape = logits.shape[:-1]
        vocab_size = logits.shape[-1]
        logits = logits.reshape(-1, vocab_size)
        if temperature == 0.0 and top_k in (None, 0) and (top_p is None or top_p >= 1.0):
            probs = F.softmax(logits, dim=-1)
            token = logits.argmax(dim=-1, keepdim=True)
            token_prob = probs.gather(-1, token)
            return (token.view(*orig_shape), token_prob.view(*orig_shape))
        if temperature > 0 and temperature != 1.0:
            logits = logits / temperature
        logits = self._top_k_logits(logits, top_k)
        logits = self._top_p_logits(logits, top_p)
        probs = F.softmax(logits, dim=-1)
        token = torch.multinomial(probs, num_samples=1)
        token_prob = torch.gather(probs, -1, token)
        return (token.view(*orig_shape), token_prob.view(*orig_shape))

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        temperature: int = 0.0,
        block_length: int = 32,
        steps: int = 32,
        gen_length: int = 2048,
        top_p: Optional[int] = None,
        top_k: Optional[int] = None,
        eos_early_stop: bool = False,
        minimal_topk: int = 1,
        threshold: float = 0.95,
        eos_id: int = 156892,
        mask_id: int = 156895,
    ):
        steps = min(steps, gen_length // minimal_topk)
        input_ids = inputs.to(self.device)
        prompt_length = input_ids.shape[1]
        num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
        total_length = num_blocks * block_length
        block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=self.device))
        block_diffusion_attention_mask = (
            block_mask.repeat_interleave(block_length, dim=0)
            .repeat_interleave(block_length, dim=1)
            .unsqueeze(0)
            .unsqueeze(0)
            .bool()
        )
        block_diffusion_attention_mask = torch.where(block_diffusion_attention_mask, 0.0, float("-inf")).to(
            torch.bfloat16
        )
        position_ids = torch.arange(total_length, device=self.device).unsqueeze(0)
        x = torch.full((1, total_length), mask_id, dtype=torch.long, device=self.device)
        x[:, :prompt_length] = input_ids.clone()
        prompt_index_full = torch.zeros_like(x, dtype=torch.bool)
        prompt_index_full[:, :prompt_length] = True
        prefill_blocks = prompt_length // block_length
        denoising_steps_per_block = steps
        num_transfer_tokens_schedule = self._get_num_transfer_tokens(block_length, denoising_steps_per_block)
        for num_block in range(prefill_blocks, num_blocks):
            current_window_end = (num_block + 1) * block_length
            cur_x = x[:, :current_window_end]
            cur_attn_mask = block_diffusion_attention_mask[:, :, :current_window_end, :current_window_end]
            cur_position_ids = position_ids[:, :current_window_end]
            for step in range(denoising_steps_per_block):
                active_block_mask = cur_x[:, -block_length:] == mask_id
                if active_block_mask.sum() == 0:
                    break
                logits = self.forward(
                    cur_x, attention_mask=cur_attn_mask, position_ids=cur_position_ids, use_cache=False
                ).logits
                active_logits = logits[:, -block_length:, :]
                x0, x0_p = self._sample_with_temperature_topk_topp(
                    active_logits, temperature=temperature, top_k=top_k, top_p=top_p
                )
                num_to_transfer = num_transfer_tokens_schedule[step].item()
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                confidence = torch.where(active_block_mask, x0_p, -torch.inf)
                high_conf_mask = confidence[0] > threshold
                num_high_confidence = high_conf_mask.sum().item()
                if num_high_confidence >= num_to_transfer:
                    transfer_index[0] = high_conf_mask
                else:
                    _, idx = torch.topk(confidence[0], k=min(num_to_transfer, active_block_mask.sum().item()))
                    transfer_index[0, idx] = True
                if transfer_index.any():
                    cur_x[:, -block_length:][transfer_index] = x0[transfer_index]
                if eos_early_stop and (x0[transfer_index] == eos_id).any():
                    eos_pos_in_x = (cur_x[0] == eos_id).nonzero(as_tuple=True)
                    if len(eos_pos_in_x[0]) > 0:
                        eos_pos = eos_pos_in_x[0][0].item()
                        if (cur_x[0, prompt_length:eos_pos] != mask_id).all():
                            final_x = x[:, :total_length][:, : eos_pos + 1]
                            return final_x
            x[:, :current_window_end] = cur_x
            if eos_id is not None and (x[0, prompt_length:current_window_end] == eos_id).any():
                break
        generated_answer = x[:, : prompt_length + gen_length]
        mask_positions = (
            (generated_answer[0][input_ids.shape[1] :] == eos_id).nonzero(as_tuple=True)[0]
            if eos_id is not None
            else torch.empty(0, dtype=torch.long)
        )
        if len(mask_positions) > 0:
            first_mask_position = mask_positions[0].item()
        else:
            first_mask_position = gen_length
        return generated_answer[:, : input_ids.shape[1] + first_mask_position + 1]


class LLaDA20CAPGeneration(LLaDA20Generation):
    def _sample_with_temperature_topk_topp(self, logits, temperature=1.0, top_k=0, top_p=1.0):
        orig_shape = logits.shape[:-1]
        vocab_size = logits.shape[-1]
        logits = logits.reshape(-1, vocab_size)
        if temperature > 0 and temperature != 1.0:
            logits = logits / temperature
        logits = self._top_k_logits(logits, top_k)
        logits = self._top_p_logits(logits, top_p)
        probs = F.softmax(logits, dim=-1)
        token = torch.multinomial(probs, num_samples=1)
        token_prob = torch.gather(probs, -1, token)
        return (token.view(*orig_shape), token_prob.view(*orig_shape))

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        temperature: int = 0.0,
        block_length: int = 32,
        steps: int = 32,
        gen_length: int = 2048,
        top_p: Optional[int] = None,
        top_k: Optional[int] = None,
        eos_early_stop: bool = False,
        minimal_topk: int = 1,
        threshold: float = 0.95,
        eos_id: int = 156892,
        mask_id: int = 156895,
    ):
        steps = min(steps, gen_length // minimal_topk)
        input_ids = inputs.to(self.device)
        prompt_length = input_ids.shape[1]
        num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
        total_length = num_blocks * block_length
        block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=self.device))
        block_diffusion_attention_mask = (
            block_mask.repeat_interleave(block_length, dim=0)
            .repeat_interleave(block_length, dim=1)
            .unsqueeze(0)
            .unsqueeze(0)
            .bool()
        )
        block_diffusion_attention_mask = torch.where(block_diffusion_attention_mask, 0.0, float("-inf")).to(
            torch.bfloat16
        )
        position_ids = torch.arange(total_length, device=self.device).unsqueeze(0)
        x = torch.full((1, total_length), mask_id, dtype=torch.long, device=self.device)
        x[:, :prompt_length] = input_ids.clone()
        prompt_index_full = torch.zeros_like(x, dtype=torch.bool)
        prompt_index_full[:, :prompt_length] = True
        prefill_blocks = prompt_length // block_length
        denoising_steps_per_block = steps
        num_transfer_tokens_schedule = self._get_num_transfer_tokens(block_length, denoising_steps_per_block)
        for num_block in range(prefill_blocks, num_blocks):
            current_window_end = (num_block + 1) * block_length
            cur_x = x[:, :current_window_end]
            cur_attn_mask = block_diffusion_attention_mask[:, :, :current_window_end, :current_window_end]
            cur_position_ids = position_ids[:, :current_window_end]
            for step in range(denoising_steps_per_block):
                active_block_mask = cur_x[:, -block_length:] == mask_id
                if active_block_mask.sum() == 0:
                    break
                logits = self.forward(
                    cur_x, attention_mask=cur_attn_mask, position_ids=cur_position_ids, use_cache=False
                ).logits
                active_logits = logits[:, -block_length:, :]
                x0, x0_p = self._sample_with_temperature_topk_topp(
                    active_logits, temperature=temperature, top_k=top_k, top_p=top_p
                )
                num_to_transfer = num_transfer_tokens_schedule[step].item()
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                confidence = torch.where(active_block_mask, x0_p, -torch.inf)
                high_conf_mask = confidence[0] > threshold
                num_high_confidence = high_conf_mask.sum().item()
                if num_high_confidence >= num_to_transfer:
                    transfer_index[0] = high_conf_mask
                else:
                    _, idx = torch.topk(confidence[0], k=min(num_to_transfer, active_block_mask.sum().item()))
                    transfer_index[0, idx] = True
                if transfer_index.any():
                    cur_x[:, -block_length:][transfer_index] = x0[transfer_index]
                if eos_early_stop and (x0[transfer_index] == eos_id).any():
                    eos_pos_in_x = (cur_x[0] == eos_id).nonzero(as_tuple=True)
                    if len(eos_pos_in_x[0]) > 0:
                        eos_pos = eos_pos_in_x[0][0].item()
                        if (cur_x[0, prompt_length:eos_pos] != mask_id).all():
                            final_x = x[:, :total_length][:, : eos_pos + 1]
                            return final_x
            x[:, :current_window_end] = cur_x
            if eos_id is not None and (x[0, prompt_length:current_window_end] == eos_id).any():
                break
        generated_answer = x[:, : prompt_length + gen_length]
        mask_positions = (
            (generated_answer[0][input_ids.shape[1] :] == eos_id).nonzero(as_tuple=True)[0]
            if eos_id is not None
            else torch.empty(0, dtype=torch.long)
        )
        if len(mask_positions) > 0:
            first_mask_position = mask_positions[0].item()
        else:
            first_mask_position = gen_length
        return generated_answer[:, input_ids.shape[1] : input_ids.shape[1] + first_mask_position + 1]


class LLaDA21Generation(LLaDA20Generation):
    def _sample_with_temperature_topk_topp(self, logits, temperature=1.0, top_k=0, top_p=1.0):
        orig_shape = logits.shape[:-1]
        vocab_size = logits.shape[-1]
        logits = logits.reshape(-1, vocab_size)
        if temperature == 0.0:
            token = torch.argmax(logits, dim=-1, keepdim=True)
            probs = F.softmax(logits, dim=-1)
            token_prob = torch.gather(probs, -1, token)
            return (token.view(*orig_shape), token_prob.view(*orig_shape))
        if temperature > 0 and temperature != 1.0:
            logits = logits / temperature
        logits = self._top_k_logits(logits, top_k)
        logits = self._top_p_logits(logits, top_p)
        probs = F.softmax(logits, dim=-1)
        token = torch.multinomial(probs, num_samples=1)
        token_prob = torch.gather(probs, -1, token)
        return (token.view(*orig_shape), token_prob.view(*orig_shape))

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        temperature: float = 0.0,
        block_length: int = 32,
        steps: int = 32,
        gen_length: int = 2048,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        eos_early_stop: bool = False,
        minimal_topk: int = 1,
        threshold: float = 0.95,
        editing_threshold: float = 0.9,
        max_post_steps: int = 16,
        eos_id: int = 156892,
        mask_id: int = 156895,
        num_to_transfer: int = 1,
        max_steps_per_block: int = 1000,
    ):
        steps = min(steps, gen_length // minimal_topk)
        input_ids = inputs.to(self.device)
        prompt_length = input_ids.shape[1]
        num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
        total_length = num_blocks * block_length
        block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=self.device))
        block_diffusion_attention_mask = (
            block_mask.repeat_interleave(block_length, dim=0)
            .repeat_interleave(block_length, dim=1)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(torch.bfloat16)
        )
        position_ids = torch.arange(total_length, device=self.device).unsqueeze(0)
        x = torch.full((1, total_length), mask_id, dtype=torch.long, device=self.device)
        x[:, :prompt_length] = input_ids.clone()
        prompt_index_full = torch.zeros_like(x, dtype=torch.bool)
        prompt_index_full[:, :prompt_length] = True
        prefill_blocks = prompt_length // block_length
        for num_block in range(prefill_blocks, num_blocks):
            current_window_end = (num_block + 1) * block_length
            cur_x = x[:, :current_window_end]
            cur_attn_mask = block_diffusion_attention_mask[:, :, :current_window_end, :current_window_end]
            cur_position_ids = position_ids[:, :current_window_end]
            block_start_pos = num_block * block_length
            post_steps = 0
            block_steps = 0
            while True:
                old_block_tokens = cur_x[:, -block_length:].clone()
                active_block_mask = cur_x[:, -block_length:] == mask_id
                if not torch.any(active_block_mask):
                    post_steps += 1
                if post_steps > max_post_steps:
                    break
                if block_steps >= max_steps_per_block:
                    raise RuntimeError(
                        "LLaDA2.1 block refinement exceeded max_steps_per_block; retry with a larger budget"
                    )
                block_steps += 1
                prompt_mask_in_block = torch.zeros(block_length, dtype=torch.bool, device=self.device)
                if block_start_pos < prompt_length:
                    prompt_end_in_block = min(prompt_length - block_start_pos, block_length)
                    prompt_mask_in_block[:prompt_end_in_block] = True
                outputs = self.forward(
                    cur_x,
                    attention_mask=cur_attn_mask,
                    position_ids=cur_position_ids,
                    use_cache=False,
                    output_attentions=True,
                )
                logits = outputs.logits
                active_logits = logits[:, -block_length:, :]
                x0, x0_p = self._sample_with_temperature_topk_topp(
                    active_logits, temperature=temperature, top_k=top_k, top_p=top_p
                )
                mask_transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                if active_block_mask.sum() > 0:
                    mask_confidence = torch.where(active_block_mask, x0_p, -torch.inf)
                    high_conf_mask = (mask_confidence[0] > threshold) & active_block_mask[0]
                    num_high_confidence = high_conf_mask.sum().item()
                    if num_high_confidence >= num_to_transfer:
                        mask_transfer_index[0] = high_conf_mask
                    else:
                        num_available = active_block_mask.sum().item()
                        if num_available > 0:
                            _, idx = torch.topk(mask_confidence[0], k=min(num_to_transfer, num_available))
                            mask_transfer_index[0, idx] = True
                editing_transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                non_mask_positions = ~active_block_mask
                non_prompt_positions = ~prompt_mask_in_block
                editable_positions = non_mask_positions & non_prompt_positions[None, :]
                editing_confidence = torch.where(editable_positions, x0_p, -torch.inf)
                high_conf_editing = (editing_confidence[0] > editing_threshold) & editable_positions[0]
                token_changed = x0[0] != old_block_tokens[0]
                editing_transfer_index[0] = high_conf_editing & token_changed
                final_transfer_index = mask_transfer_index | editing_transfer_index
                if final_transfer_index.any():
                    cur_x[:, -block_length:][final_transfer_index] = x0[final_transfer_index]
                if active_block_mask.sum() == 0 and (not editing_transfer_index.any()):
                    break
            x[:, :current_window_end] = cur_x
            if eos_early_stop:
                generated_part = x[0, prompt_length:current_window_end]
                if (generated_part == mask_id).sum() == 0:
                    eos_positions = (generated_part == eos_id).nonzero(as_tuple=True)[0]
                    if len(eos_positions) > 0:
                        break
        generated_answer = x[:, : prompt_length + gen_length]
        mask_positions = (
            (generated_answer[0][input_ids.shape[1] :] == eos_id).nonzero(as_tuple=True)[0]
            if eos_id is not None
            else torch.empty(0, dtype=torch.long)
        )
        if len(mask_positions) > 0:
            first_mask_position = mask_positions[0].item()
        else:
            first_mask_position = gen_length
        return generated_answer[:, input_ids.shape[1] : input_ids.shape[1] + first_mask_position + 1]


class LLaDA22Generation(LLaDA20Generation):
    def _sample_with_temperature_topk_topp(self, logits, temperature=1.0, top_k=0, top_p=1.0):
        orig_shape = logits.shape[:-1]
        vocab_size = logits.shape[-1]
        logits = logits.reshape(-1, vocab_size)
        if temperature > 0 and temperature != 1.0:
            logits = logits / temperature
        logits = self._top_k_logits(logits, top_k)
        logits = self._top_p_logits(logits, top_p)
        probs = F.softmax(logits, dim=-1)
        token = torch.multinomial(probs, num_samples=1)
        token_prob = torch.gather(probs, -1, token)
        return (token.view(*orig_shape), token_prob.view(*orig_shape))

    @staticmethod
    def _apply_edit_operations_with_tracking(
        block_tokens,
        old_block_tokens,
        is_original_mask_snapshot,
        mask_id,
        block_length,
        delete_token_id,
        split_token_id,
    ):
        result_tokens = []
        result_tracking = []
        for i, token in enumerate(block_tokens):
            if token == delete_token_id:
                continue
            elif token == split_token_id:
                old_token = old_block_tokens[i] if i < len(old_block_tokens) else mask_id
                snapshot_val = is_original_mask_snapshot[i] if i < len(is_original_mask_snapshot) else False
                result_tokens.extend([mask_id, old_token])
                result_tracking.extend([False, snapshot_val])
            else:
                snapshot_val = is_original_mask_snapshot[i] if i < len(is_original_mask_snapshot) else False
                result_tokens.append(token)
                result_tracking.append(snapshot_val)
        if len(result_tokens) > block_length:
            result_tokens = result_tokens[:block_length]
            result_tracking = result_tracking[:block_length]
        elif len(result_tokens) < block_length:
            pad_count = block_length - len(result_tokens)
            result_tokens.extend([mask_id] * pad_count)
            result_tracking.extend([False] * pad_count)
        return (result_tokens, result_tracking)

    def _diffusion_sample(self, logits, temperature=1.0, top_k=None, top_p=None):
        orig_shape = logits.shape[:-1]
        vocab_size = logits.shape[-1]
        logits = logits.reshape(-1, vocab_size)
        probs_full = F.softmax(logits, dim=-1)
        if temperature is None or temperature == 0.0:
            token = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            scaled = logits / temperature if temperature != 1.0 else logits
            scaled = self._top_k_logits(scaled, top_k or 0)
            scaled = self._top_p_logits(scaled, top_p if top_p is not None else 1.0)
            probs = F.softmax(scaled, dim=-1)
            token = torch.multinomial(probs, num_samples=1)
        token_prob = torch.gather(probs_full, -1, token)
        return (token.view(*orig_shape), token_prob.view(*orig_shape))

    def _resample_to_escape_loop(
        self,
        block_ids,
        old_block_ids,
        block_logits,
        mt2_index,
        t2t_index,
        seen_block_results,
        temperature,
        top_k,
        top_p,
        max_iters=5,
    ):
        if tuple(block_ids.tolist()) not in seen_block_results:
            return
        for _ in range(max_iters):
            changed_positions = (block_ids != old_block_ids).nonzero(as_tuple=True)[0].tolist()
            if not changed_positions:
                break
            rand_idx = torch.randint(len(changed_positions), (1,), device=block_ids.device).item()
            pos = changed_positions[rand_idx]
            current_token = block_ids[pos].item()
            pos_logits = block_logits[pos, :].clone()
            pos_logits[current_token] = float("-inf")
            if bool(mt2_index[pos]):
                new_token, _ = self._diffusion_sample(
                    pos_logits.unsqueeze(0), temperature=temperature, top_k=top_k, top_p=top_p
                )
            elif bool(t2t_index[pos]):
                new_token, _ = self._diffusion_sample(pos_logits.unsqueeze(0), temperature=0.0, top_k=None, top_p=None)
            else:
                continue
            block_ids[pos] = new_token.view(-1)[0]
            if tuple(block_ids.tolist()) not in seen_block_results:
                break

    @torch.no_grad()
    def _joint_decode_block(
        self,
        x,
        block_start,
        block_end,
        attention_mask,
        position_ids,
        temperature,
        top_k,
        top_p,
        steps,
        threshold,
        editing_threshold,
        max_post_steps,
        mask_id,
        delete_token_id,
        split_token_id,
        max_steps_per_block,
    ):
        device = x.device
        block_length = block_end - block_start
        prompt_mask_block = (x[0, block_start:block_end] != mask_id).clone()
        block_ids = x[0, block_start:block_end]
        prompt_edit_tokens = prompt_mask_block & ((block_ids == delete_token_id) | (block_ids == split_token_id))
        if prompt_edit_tokens.any():
            logger.warning_once(
                "Reserved edit token(s) found in a prompt/context segment of block@%d; they will be deleted/expanded by the edit-op pass.",
                block_start,
            )
        is_original_mask = (x[0, block_start:block_end] == mask_id).tolist()
        initial_mask_count = sum(is_original_mask)
        if initial_mask_count == 0:
            return
        transfer_schedule = self._get_num_transfer_tokens(initial_mask_count, steps)
        seen_block_results = set()
        post_steps = 0
        step_id = 0
        while True:
            blk = x[0, block_start:block_end]
            old_block = blk.clone()
            input_key = tuple(old_block.tolist())
            mask_index = (old_block == mask_id) & ~prompt_mask_block
            original_mask_count = sum(
                (1 for i, v in enumerate(is_original_mask) if v and old_block[i].item() == mask_id)
            )
            new_mask_count = mask_index.sum().item() - original_mask_count
            if original_mask_count == 0:
                post_steps += 1
            else:
                post_steps = 0
            logger.debug(
                "block@%d step=%d original_masks=%d new_masks=%d post_steps=%d/%d",
                block_start,
                step_id,
                original_mask_count,
                new_mask_count,
                post_steps,
                max_post_steps,
            )
            if mask_index.sum().item() == 0 and post_steps > max_post_steps:
                logger.debug(
                    "block@%d terminating: max_post_steps (%d) exceeded after %d steps",
                    block_start,
                    max_post_steps,
                    step_id,
                )
                break
            if step_id >= max_steps_per_block:
                logger.debug(
                    "block@%d terminating: max_steps_per_block (%d) reached", block_start, max_steps_per_block
                )
                break
            logits = self.forward(
                x[:, :block_end], attention_mask=attention_mask, position_ids=position_ids, use_cache=False
            ).logits
            block_logits = logits[0, block_start:block_end, :]
            x_s, p_s = self._diffusion_sample(block_logits, temperature, top_k, top_p)
            if temperature != 0.0:
                x0, p0 = self._diffusion_sample(block_logits, 0.0, None, None)
            else:
                x0, p0 = (x_s, p_s)
            neg_inf = torch.full_like(p0, -float("inf"))
            mt2_index = torch.zeros(block_length, dtype=torch.bool, device=device)
            if mask_index.any():
                if step_id < len(transfer_schedule):
                    num_need = transfer_schedule[step_id].item() + new_mask_count
                    mask_conf = torch.where(mask_index, p_s, neg_inf)
                    high_conf = (mask_conf > threshold) & mask_index
                    if high_conf.sum().item() >= num_need:
                        mt2_index = high_conf
                    else:
                        k_val = min(num_need, mask_index.sum().item())
                        if k_val > 0:
                            _, idx = torch.topk(mask_conf, k=k_val)
                            mt2_index[idx] = True
                else:
                    mt2_index = mask_index
            editable_mask = ~mask_index & ~prompt_mask_block
            editing_confidence = torch.where(editable_mask, p0, neg_inf)
            high_conf_edit = (editing_confidence > editing_threshold) & editable_mask
            token_changed = old_block != x0
            t2t_index = high_conf_edit & token_changed
            fill_index_pre = mt2_index | t2t_index
            if max_post_steps > 0:
                final_round = post_steps >= max_post_steps
            else:
                remaining_original = mask_index & torch.tensor(is_original_mask, device=device)
                final_round = remaining_original.any() and (remaining_original <= mt2_index).all()
            if final_round:
                mt2_index = mask_index
                fill_index_pre = mt2_index | t2t_index
                m2t_sd = mt2_index & ((x_s == split_token_id) | (x_s == delete_token_id))
                if m2t_sd.any():
                    sd_logits = block_logits[m2t_sd].clone()
                    sd_logits[:, split_token_id] = float("-inf")
                    sd_logits[:, delete_token_id] = float("-inf")
                    new_tokens, _ = self._diffusion_sample(sd_logits, temperature, top_k, top_p)
                    x_s[m2t_sd] = new_tokens
                t2t_sd = t2t_index & ((x0 == split_token_id) | (x0 == delete_token_id))
                if t2t_sd.any():
                    sd_logits = block_logits[t2t_sd].clone()
                    sd_logits[:, split_token_id] = float("-inf")
                    sd_logits[:, delete_token_id] = float("-inf")
                    new_tokens, _ = self._diffusion_sample(sd_logits, temperature=0.0, top_k=None, top_p=None)
                    x0[t2t_sd] = new_tokens
            is_original_mask_snapshot = list(is_original_mask)
            if fill_index_pre.any():
                if mt2_index.any():
                    blk[mt2_index] = x_s[mt2_index]
                if t2t_index.any():
                    blk[t2t_index] = x0[t2t_index]
                if not final_round:
                    self._resample_to_escape_loop(
                        blk,
                        old_block,
                        block_logits,
                        mt2_index,
                        t2t_index,
                        seen_block_results,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                    )
            seen_block_results.add(tuple(blk.tolist()))
            edited_block, is_original_mask = self._apply_edit_operations_with_tracking(
                blk.tolist(),
                old_block.tolist(),
                is_original_mask_snapshot,
                mask_id,
                block_length,
                delete_token_id=delete_token_id,
                split_token_id=split_token_id,
            )
            x[0, block_start:block_end] = torch.tensor(edited_block, device=device, dtype=x.dtype)
            if (
                x[0, block_start:block_end].tolist() == list(input_key)
                and (x[0, block_start:block_end] == mask_id).sum() == 0
            ):
                step_id += 1
                break
            step_id += 1

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        temperature: float = 0.0,
        block_length: int = 32,
        steps: int = 32,
        gen_length: int = 2048,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        threshold: float = 0.5,
        editing_threshold: float = 0.0,
        max_post_steps: int = 16,
        eos_early_stop: bool = False,
        eos_id: int = 156892,
        mask_id: int = 156895,
        delete_token_id: int = 156930,
        split_token_id: int = 156931,
        max_steps_per_block: int = 1000,
    ):
        input_ids = inputs.to(self.device)
        prompt_length = input_ids.shape[1]
        num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
        total_length = num_blocks * block_length
        block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=self.device))
        block_diffusion_attention_mask = (
            block_mask.repeat_interleave(block_length, dim=0)
            .repeat_interleave(block_length, dim=1)
            .unsqueeze(0)
            .unsqueeze(0)
            .log()
            .to(torch.bfloat16)
        )
        position_ids = torch.arange(total_length, device=self.device).unsqueeze(0)
        x = torch.full((1, total_length), mask_id, dtype=torch.long, device=self.device)
        x[:, :prompt_length] = input_ids.clone()
        prefill_blocks = prompt_length // block_length
        for num_block in range(prefill_blocks, num_blocks):
            block_start = num_block * block_length
            block_end = (num_block + 1) * block_length
            cur_attn_mask = block_diffusion_attention_mask[:, :, :block_end, :block_end]
            cur_position_ids = position_ids[:, :block_end]
            self._joint_decode_block(
                x,
                block_start,
                block_end,
                cur_attn_mask,
                cur_position_ids,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                steps=steps,
                threshold=threshold,
                editing_threshold=editing_threshold,
                max_post_steps=max_post_steps,
                mask_id=mask_id,
                delete_token_id=delete_token_id,
                split_token_id=split_token_id,
                max_steps_per_block=max_steps_per_block,
            )
            if eos_early_stop and eos_id is not None and (x[0, prompt_length:block_end] == eos_id).any():
                break
        generated_answer = x[:, : prompt_length + gen_length]
        eos_positions = (
            (generated_answer[0][prompt_length:] == eos_id).nonzero(as_tuple=True)[0]
            if eos_id is not None
            else torch.empty(0, dtype=torch.long)
        )
        if len(eos_positions) > 0:
            first_eos_position = eos_positions[0].item()
        else:
            first_eos_position = gen_length
        output = generated_answer[:, prompt_length : prompt_length + first_eos_position + 1]
        residual = (output == mask_id) | (output == split_token_id) | (output == delete_token_id)
        if residual.any():
            logger.warning(
                "Decoding finished with %d residual mask/split/delete token(s) in the output; the generation may be malformed.",
                int(residual.sum().item()),
            )
        return output
