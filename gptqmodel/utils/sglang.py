import multiprocessing as mp

from transformers import AutoConfig

try:
    import sglang as sgl
    SGLANG_AVAILABLE = True
except ImportError:
    SGLANG_AVAILABLE = False

SGLANG_INSTALL_HINT = "sglang not installed. Please install via `pip install -U sglang`."

def load_model_by_sglang(
    model,
    trust_remote_code,
    **kwargs
):
    if not SGLANG_AVAILABLE:
        raise ValueError(SGLANG_INSTALL_HINT)

    mp.set_start_method('spawn')
    runtime = sgl.Runtime(
        model_path=model,
        **kwargs,
    )
    sgl.set_default_backend(runtime)
    hf_config = AutoConfig.from_pretrained(
        model, trust_remote_code=trust_remote_code
    )
    return runtime, hf_config

@sgl.function
def generate(s, prompt, **kwargs):
    s += prompt
    s += sgl.gen(
        "result",
        **kwargs,
    )

def sglang_generate(
        **kwargs,
):
    if not SGLANG_AVAILABLE:
        raise ValueError(SGLANG_INSTALL_HINT)

    prompts = kwargs.pop("prompts", None)
    state = generate.run(
        prompt=prompts,
        **kwargs,
    )

    return state["result"]
