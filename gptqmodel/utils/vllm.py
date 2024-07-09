
from vllm import LLM, SamplingParams

def load_model_by_vllm(
    model,
    trust_remote_code,
    **kwargs,
):
    model = LLM(
        model=model,
        trust_remote_code=trust_remote_code,
        **kwargs,
    )

    return model

def vllm_generate(
        model,
        **kwargs,
):
    prompts = kwargs.pop("prompts", None)
    sampling_params = kwargs.pop("sampling_params", None)
    outputs = model.generate(prompts, sampling_params)
    return outputs