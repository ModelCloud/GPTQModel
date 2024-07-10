import multiprocessing as mp

from transformers import AutoConfig


def load_model_by_sglang(
    model,
    trust_remote_code,
    **kwargs
):
    from sglang.srt.server import Runtime
    mp.set_start_method('spawn')
    runtime = Runtime(
        model_path=model,
        **kwargs,
    )

    hf_config = AutoConfig.from_pretrained(
        model, trust_remote_code=trust_remote_code
    )
    return runtime, hf_config

async def sglang_generate(
        model,
        **kwargs,
):

    prompts = kwargs.pop("prompts", None)
    sampling_params = kwargs.pop("sampling_params", None)

    if sampling_params is None:
        sampling_params = {key: kwargs[key] for key in [
            'repetition_penalty', 'temperature',
            'top_k', 'top_p'
        ] if key in kwargs}
    stream = model.add_request(prompts, sampling_params)

    return stream
