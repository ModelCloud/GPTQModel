# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

from typing import Any, Dict

import torch


try:
    from vllm import LLM, SamplingParams, TokensPrompt

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

VLLM_INSTALL_HINT = "vLLM not installed. Please install via `pip install -U vllm`."

# returns SamplingParams but we can't use this typehint since vLLM is optional depend
def convert_hf_params_to_vllm(hf_params: Dict[str, Any]):
    if not VLLM_AVAILABLE:
        raise ValueError(VLLM_INSTALL_HINT)
    sampling_params = SamplingParams()

    if hf_params.get('num_return_sequences', None):
        sampling_params.n = hf_params.get('num_return_sequences')

    if hf_params.get('repetition_penalty', None):
        sampling_params.repetition_penalty = hf_params.get('repetition_penalty')

    if hf_params.get('temperature', None):
        sampling_params.temperature = hf_params.get('temperature')

    if hf_params.get('top_k', None):
        sampling_params.top_k = hf_params.get('top_k')

    if hf_params.get('top_p', None):
        sampling_params.top_p = hf_params.get('top_p')

    if hf_params.get('max_length', None):
        raise ValueError("vLLM does not support argument `max_length`. Please use `max_new_tokens` instead.")
    if hf_params.get('min_length', None):
        raise ValueError("vLLM does not support argument `min_length`. Please use `min_new_tokens` instead.")

    if hf_params.get('max_new_tokens', None):
        sampling_params.max_tokens = hf_params.get('max_new_tokens')

    if hf_params.get('min_new_tokens', None):
        sampling_params.min_tokens = hf_params.get('min_new_tokens')

    if hf_params.get('eos_token_id', None):
        sampling_params.stop_token_ids = [hf_params.get('eos_token_id'), None]

    return sampling_params


def load_model_by_vllm(
        model,
        **kwargs,
):
    if not VLLM_AVAILABLE:
        raise ValueError(VLLM_INSTALL_HINT)

    model = LLM(
        model=model,
        **kwargs,
    )

    return model


@torch.inference_mode()
def vllm_generate(model, **kwargs):
    if not VLLM_AVAILABLE:
        raise ValueError(VLLM_INSTALL_HINT)

    # Extract and validate prompts
    prompts = kwargs.pop("prompts", None) or kwargs.pop("input_ids", None)
    if prompts is None:
        raise ValueError("Either prompts or input_ids must be provided")

    sampling_params = kwargs.get("sampling_params")
    if not isinstance(sampling_params, SamplingParams):
        hf_params = {
            key: kwargs.get(key) for key in [
                'num_return_sequences', 'repetition_penalty', 'temperature',
                'top_k', 'top_p', 'max_length', 'min_length', 'max_new_tokens', 'min_new_tokens', 'eos_token_id'
            ]
        }
        sampling_params = convert_hf_params_to_vllm({k: v for k, v in hf_params.items() if v is not None})

    # Convert prompts to vLLM format
    if isinstance(prompts, torch.Tensor):
        tokens_prompts = [TokensPrompt(prompt_token_ids=prompt) for prompt in prompts.tolist()]
        req_results = model.generate(prompts=tokens_prompts, sampling_params=sampling_params)
    elif isinstance(prompts, list):
        if isinstance(prompts[0], list) or isinstance(prompts[0], int):
            tokens_prompts = [TokensPrompt(prompt_token_ids=prompt) for prompt in prompts]
            req_results = model.generate(prompts=tokens_prompts, sampling_params=sampling_params)
        else:
            req_results = model.generate(prompts=prompts, sampling_params=sampling_params)
    elif isinstance(prompts, str):
        req_results = model.generate(prompts=prompts, sampling_params=sampling_params)
    else:
        raise ValueError(f"Invalid input type for vllm_generate, type is {type(prompts)}")

    outputs = []
    for result in req_results:
        combined_token_ids = result.prompt_token_ids + list(result.outputs[0].token_ids)
        outputs.append(combined_token_ids)

    pad_token_id = model.get_tokenizer().pad_token_id
    if pad_token_id is None:
        pad_token_id = model.get_tokenizer().eos_token_id
    max_length = max(len(sublist) for sublist in outputs)
    padded_list = [sublist + [pad_token_id] * (max_length - len(sublist)) for sublist in outputs]

    return torch.Tensor(padded_list).to(torch.uint32)
