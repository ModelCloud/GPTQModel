# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence

import torch
from transformers import StaticCache


class StaticCUDAGraphGreedyRunner:
    """Reusable batch-one Qwen3 greedy decode graph for a fixed token and cache capacity.

    This is an explicit low-level path. It supports raw token IDs, greedy argmax, full-attention Qwen3 models,
    and one CUDA stream. Sampling, logits processors, padding masks, beam search, streaming, and concurrent use
    should continue to use ``model.generate``. Do not move the model or clear its weight caches after capture.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        example_input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        max_cache_len: int | None = None,
        graph_warmup: int = 2,
        release_prefill_cache: bool = True,
        capture_prefill: bool = False,
    ) -> None:
        self._validate_model_and_inputs(model, example_input_ids)
        if max_new_tokens < 2:
            raise ValueError("max_new_tokens must be at least 2 when capturing a decode graph.")
        if graph_warmup < 1:
            raise ValueError("graph_warmup must be at least 1 so decode kernels are initialized before capture.")
        if capture_prefill and release_prefill_cache:
            raise ValueError("capture_prefill=True requires release_prefill_cache=False to keep graph pointers live.")

        required_cache_len = example_input_ids.shape[1] + max_new_tokens
        max_cache_len = required_cache_len if max_cache_len is None else max_cache_len
        if max_cache_len < required_cache_len:
            raise ValueError(
                f"max_cache_len must be at least {required_cache_len} for the example prompt and token capacity; "
                f"actual value is {max_cache_len}."
            )

        max_positions = getattr(model.config, "max_position_embeddings", None)
        if max_positions is not None and max_cache_len > max_positions:
            raise ValueError(
                f"max_cache_len={max_cache_len} exceeds model.config.max_position_embeddings={max_positions}."
            )

        self.model = model
        self.device = example_input_ids.device
        self.max_new_tokens = max_new_tokens
        self.max_cache_len = max_cache_len
        self.release_prefill_cache = release_prefill_cache
        self.capture_prefill = capture_prefill
        self.prompt_tokens = example_input_ids.shape[1]
        self._execution_stream = torch.cuda.current_stream(self.device)
        self._cache = StaticCache(config=model.config, max_cache_len=max_cache_len)
        self._install_prism_q2_cache(self._cache)
        self._token = torch.empty((1, 1), device=self.device, dtype=torch.long)
        self._position = torch.empty((1, 1), device=self.device, dtype=torch.long)
        self._generated = torch.empty((1, max_new_tokens), device=self.device, dtype=torch.long)
        self._capture(example_input_ids, graph_warmup=graph_warmup)

    @staticmethod
    def _validate_model_and_inputs(model: torch.nn.Module, input_ids: torch.Tensor) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("StaticCUDAGraphGreedyRunner requires CUDA; use model.generate on other devices.")
        if model.training:
            raise ValueError("StaticCUDAGraphGreedyRunner requires model.eval().")
        if getattr(getattr(model, "config", None), "model_type", None) != "qwen3":
            raise ValueError("StaticCUDAGraphGreedyRunner currently supports Qwen3 decoder-only models.")
        if not torch.is_tensor(input_ids) or input_ids.dtype != torch.long:
            raise TypeError("input_ids must be a torch.long tensor.")
        if input_ids.device.type != "cuda":
            raise ValueError("input_ids must be on the same CUDA device as the model.")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must have shape [1, prompt_tokens] with a non-empty prompt.")

        embedding = model.get_input_embeddings()
        if embedding.weight.device != input_ids.device:
            raise ValueError("input_ids must be on the same CUDA device as the model embeddings.")
        if embedding.weight.dtype != torch.float16:
            raise ValueError("StaticCUDAGraphGreedyRunner currently requires FP16 model activations.")

        layer_types = getattr(model.config, "layer_types", None)
        if layer_types is not None and any(layer_type != "full_attention" for layer_type in layer_types):
            raise ValueError("StaticCUDAGraphGreedyRunner currently requires full-attention cache layers.")

    def _model_step(self, token: torch.Tensor, position: torch.Tensor, cache: StaticCache):
        return self.model(
            input_ids=token,
            position_ids=position,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )

    def _install_prism_q2_cache(self, cache: StaticCache) -> None:
        from ..nn_modules.triton_utils.q2_cache import install_prism_q2_static_cache

        install_prism_q2_static_cache(self.model, cache, device=self.device)

    def _prefill(self, input_ids: torch.Tensor):
        positions = torch.arange(input_ids.shape[1], device=self.device).unsqueeze(0)
        return self.model(
            input_ids=input_ids,
            position_ids=positions,
            past_key_values=self._cache,
            use_cache=True,
            logits_to_keep=1,
        )

    def _release_prefill_weight_cache(self) -> None:
        if not self.release_prefill_cache:
            return
        for module in self.model.modules():
            release = getattr(module, "release_q2_prefill_cache", None)
            if callable(release):
                release()

    def _collect_cache_lengths(self) -> None:
        self._cache_lengths = []
        for layer in self._cache.layers:
            cumulative_length = getattr(layer, "cumulative_length", None)
            if not torch.is_tensor(cumulative_length) or getattr(layer, "is_sliding", False):
                raise RuntimeError("Captured cache does not contain the expected full-attention static layers.")
            self._cache_lengths.append(cumulative_length)

    def _pin_prefill_weight_cache(self) -> None:
        self._prefill_weight_tensors = []
        for module in self.model.modules():
            caches = getattr(module, "_gguf_triton_cache", None)
            if not isinstance(caches, dict):
                continue
            for cache in caches.values():
                self._prefill_weight_tensors.extend(value for value in cache.values() if torch.is_tensor(value))

    def _capture(self, example_input_ids: torch.Tensor, *, graph_warmup: int) -> None:
        with torch.inference_mode():
            output = self._prefill(example_input_ids)
            self._token.copy_(output.logits[:, -1:, :].argmax(dim=-1))
            self._position.fill_(example_input_ids.shape[1])
            self._collect_cache_lengths()

            capture_stream = torch.cuda.Stream(device=self.device)
            if graph_warmup:
                scratch_cache = StaticCache(config=self.model.config, max_cache_len=self.max_cache_len)
                self._install_prism_q2_cache(scratch_cache)
                scratch_token = self._token.clone()
                scratch_position = torch.zeros_like(self._position)
                capture_stream.wait_stream(self._execution_stream)
                with torch.cuda.stream(capture_stream):
                    for _ in range(graph_warmup):
                        scratch_output = self._model_step(scratch_token, scratch_position, scratch_cache)
                        scratch_token.copy_(scratch_output.logits[:, -1:, :].argmax(dim=-1))
                        scratch_position.add_(1)
                self._execution_stream.wait_stream(capture_stream)
                torch.cuda.synchronize(self.device)
                del scratch_cache, scratch_output, scratch_position, scratch_token

            graph_pool = torch.cuda.graph_pool_handle() if self.capture_prefill else None
            if self.capture_prefill:
                self._pin_prefill_weight_cache()
                self._static_input_ids = torch.empty_like(example_input_ids)
                self._static_input_ids.copy_(example_input_ids)
                self._prefill_positions = torch.arange(self.prompt_tokens, device=self.device).unsqueeze(0)
                self._reset_cache_length()
                capture_stream.wait_stream(self._execution_stream)
                self._prefill_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self._prefill_graph, stream=capture_stream, pool=graph_pool):
                    self._prefill_graph_output = self.model(
                        input_ids=self._static_input_ids,
                        position_ids=self._prefill_positions,
                        past_key_values=self._cache,
                        use_cache=True,
                        logits_to_keep=1,
                    )
                    self._token.copy_(self._prefill_graph_output.logits[:, -1:, :].argmax(dim=-1))
                    self._position.fill_(self.prompt_tokens)
                self._execution_stream.wait_stream(capture_stream)
                torch.cuda.synchronize(self.device)
            else:
                self._release_prefill_weight_cache()

            capture_stream.wait_stream(self._execution_stream)
            self._graph = torch.cuda.CUDAGraph()
            graph_kwargs = {"pool": graph_pool} if graph_pool is not None else {}
            with torch.cuda.graph(self._graph, stream=capture_stream, **graph_kwargs):
                self._graph_output = self._model_step(self._token, self._position, self._cache)
                self._token.copy_(self._graph_output.logits[:, -1:, :].argmax(dim=-1))
                self._position.add_(1)
            self._execution_stream.wait_stream(capture_stream)
            torch.cuda.synchronize(self.device)

    def _reset_cache_length(self) -> None:
        # Invalid cache slots are masked, and every live slot is overwritten before it is read. Avoid clearing all KV
        # tensors between requests; resetting their device-side lengths is sufficient for full-attention StaticCache.
        torch._foreach_zero_(self._cache_lengths)

    @staticmethod
    def _normalize_eos_token_ids(eos_token_id: int | Sequence[int] | None) -> tuple[int, ...]:
        if eos_token_id is None:
            return ()
        if isinstance(eos_token_id, int):
            return (eos_token_id,)
        values = tuple(int(token_id) for token_id in eos_token_id)
        if not values:
            raise ValueError("eos_token_id cannot be an empty sequence.")
        return values

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int | None = None,
        eos_token_id: int | Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Generate raw greedy token IDs, truncating the visible result at the first EOS token."""

        self._validate_model_and_inputs(self.model, input_ids)
        if torch.cuda.current_stream(self.device) != self._execution_stream:
            raise RuntimeError("StaticCUDAGraphGreedyRunner.generate must run on the stream used during capture.")

        requested_tokens = self.max_new_tokens if max_new_tokens is None else max_new_tokens
        if requested_tokens < 1 or requested_tokens > self.max_new_tokens:
            raise ValueError(f"max_new_tokens must be in [1, {self.max_new_tokens}]; actual value is {requested_tokens}.")
        if input_ids.shape[1] + requested_tokens > self.max_cache_len:
            raise ValueError(
                f"prompt plus requested tokens exceeds max_cache_len={self.max_cache_len}: "
                f"{input_ids.shape[1]} + {requested_tokens}."
            )
        if self.capture_prefill and input_ids.shape[1] != self.prompt_tokens:
            raise ValueError(
                f"Captured prefill requires exactly {self.prompt_tokens} prompt tokens; "
                f"actual value is {input_ids.shape[1]}."
            )

        self._reset_cache_length()
        if self.capture_prefill:
            self._static_input_ids.copy_(input_ids)
            self._prefill_graph.replay()
        else:
            output = self._prefill(input_ids)
            self._token.copy_(output.logits[:, -1:, :].argmax(dim=-1))
            self._position.fill_(input_ids.shape[1])
            self._release_prefill_weight_cache()
        self._generated[:, :1].copy_(self._token)
        for output_index in range(1, requested_tokens):
            self._graph.replay()
            self._generated[:, output_index : output_index + 1].copy_(self._token)

        generated = self._generated[:, :requested_tokens].clone()
        eos_token_ids = self._normalize_eos_token_ids(eos_token_id)
        if eos_token_ids:
            eos = torch.tensor(eos_token_ids, device=self.device, dtype=torch.long)
            matches = (generated.unsqueeze(-1) == eos).any(dim=-1)[0]
            locations = torch.nonzero(matches, as_tuple=False)
            if locations.numel():
                generated = generated[:, : int(locations[0].item()) + 1]
        return torch.cat((input_ids, generated), dim=1)


__all__ = ["StaticCUDAGraphGreedyRunner"]
