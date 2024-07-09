try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    
VLLM_INSTALL_HINT = "vLLM not installed. Please install via `pip install -U vllm`."

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
    if isinstance(sampling_params, SamplingParams):
        outputs = model.generate(prompts, sampling_params)
    else:
        # TODO: convert/extract HF generate params and convert into vllm.SamplingParams
        raise ValueError("Please pass in vllm.SamplingParams as `sampling_params`.")
    
    return outputs
