import logging

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
from typing import Any, Dict

VLLM_INSTALL_HINT = "vLLM not installed. Please install via `pip install -U vllm`."

def convert_hf_params_to_vllm(hf_params: Dict[str, Any]) -> SamplingParams:

    params = {
        'n': hf_params.get('num_return_sequences', 1),
        'repetition_penalty': hf_params.get('repetition_penalty', 1.0),
        'temperature': hf_params.get('temperature', 1.0),
        'top_k': hf_params.get('top_k', -1),
        'top_p': hf_params.get('top_p', 1.0),
        'max_tokens': hf_params.get('max_length', 16),
        'min_tokens': hf_params.get('min_length', 0),
        'early_stopping': hf_params.get('early_stopping', False),
        'length_penalty': hf_params.get('length_penalty', 1.0),
        'stop_token_ids': [hf_params.get('eos_token_id'), None],
    }
    return SamplingParams(**params)

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

def vllm_generate(
        model,
        **kwargs,
):
    if not VLLM_AVAILABLE:
        raise ValueError(VLLM_INSTALL_HINT)

    prompts = kwargs.pop("prompts", None)
    sampling_params = kwargs.pop("sampling_params", None)

    if not isinstance(sampling_params, SamplingParams):
        hf_params = {key: kwargs[key] for key in [
            'num_return_sequences', 'repetition_penalty', 'temperature',
            'top_k', 'top_p', 'max_length', 'min_length',
            'early_stopping', 'length_penalty', 'eos_token_id'
        ] if key in kwargs}
        sampling_params = convert_hf_params_to_vllm(hf_params)

    outputs = model.generate(prompts, sampling_params)
    return outputs
