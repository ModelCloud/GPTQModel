import sglang
from sglang.srt.server import Runtime
import optimizer
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.server import generate_request
def load_model_by_sglang(
    model,
    trust_remote_code,
):
    model = Runtime(
        model_path=model,
        tokenizer_path=model,
        # MUST install with `pip install -U /monster/data/pkg/flashinfer*whl`
        max_prefill_tokens=2048,
        enable_flashinfer=True,
        mem_fraction_static=0.8,
        disable_disk_cache=True,
        trust_remote_code=False,
        random_seed=optimizer.RAND_SEED,
        tokenizer_mode="fast",
        tp_size=1,
    )

    return model

async def sglang_generate(
        model,
        **kwargs,
):
    prompts = kwargs.pop("prompts", None)
    input_ids = model.tokenizer.encode(prompts)
    sampling_params = kwargs.pop("sampling_params", None)
    req = GenerateReqInput(input_ids=input_ids, sampling_params=sampling_params, stream=False)
    output: dict = await generate_request(req)

    return output