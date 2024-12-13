from typing import Any, Dict

import torch

try:
    from vllm import LLM, SamplingParams

    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False

VLLM_INSTALL_HINT = "vLLM not installed. Please install via `pip install -U vllm`."


# returns SamplingParams but we can't use this typehint since vLLM is optional depend
def convert_hf_params_to_vllm(hf_params: Dict[str, Any]):
    if not VLLM_AVAILABLE:
        raise ValueError(VLLM_INSTALL_HINT)

    params = {
        'n': hf_params.get('num_return_sequences', 1),
        'repetition_penalty': hf_params.get('repetition_penalty', 1.0),
        'temperature': hf_params.get('temperature', 1.0),
        'top_k': hf_params.get('top_k', -1),
        'top_p': hf_params.get('top_p', 1.0),
        'max_tokens': hf_params.get('max_length', 2048),
        'min_tokens': hf_params.get('min_length', 0),
        'stop_token_ids': [hf_params.get('eos_token_id'), [None, None]],
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
                'top_k', 'top_p', 'max_length', 'min_length', 'eos_token_id'
            ]
        }
        sampling_params = convert_hf_params_to_vllm({k: v for k, v in hf_params.items() if v is not None})

    # Convert prompts to vLLM format
    if isinstance(prompts, torch.Tensor):
        req_results = model.generate(prompt_token_ids=prompts.tolist(), sampling_params=sampling_params)
    elif isinstance(prompts, list):
        if isinstance(prompts[0], list) or isinstance(prompts[0], int):
            req_results = model.generate(prompt_token_ids=prompts, sampling_params=sampling_params)
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
