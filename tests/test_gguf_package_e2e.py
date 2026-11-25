# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""End-to-end: the packaged file must execute like the checkpoint it came from.

A byte comparison proves the container; this test proves the *model*. It
quantizes a real checkpoint, packages it, then decodes the same prompt greedily
through llama.cpp and through GPT-QModel and compares the tokens and the
next-token logits. The two engines share no kernel, so the assertion bounds are
the numerical agreement two correct implementations can reach, not bit
equality.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gptqmodel.packaging import GGUFPackageConfig, package


MODEL_ID = Path("/monster/data/model/Qwen3-0.6B")

MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital city of France? Answer in one short sentence."},
]

NEW_TOKENS = 24
#: Greedy decoding is a tie-break on logits; one flip is the most a correct
#: pair of engines may disagree on this prompt.
MAX_TOKEN_FLIPS = 1
MIN_LOGIT_COSINE = 0.999


pytestmark = [
    pytest.mark.timeout(3600),
    pytest.mark.skipif(not MODEL_ID.is_dir(), reason=f"local model fixture is unavailable: {MODEL_ID}"),
]


@pytest.fixture(scope="module")
def engines(tmp_path_factory):
    torch = pytest.importorskip("torch")
    pytest.importorskip("gguf")
    llama_cpp = pytest.importorskip("llama_cpp")
    if not torch.cuda.is_available():
        pytest.skip("the end-to-end comparison needs a CUDA device")
    from transformers import AutoTokenizer

    from gptqmodel import GPTQModel
    from gptqmodel.quantization import GGUFConfig

    root = tmp_path_factory.mktemp("gguf-package-e2e")
    checkpoint = root / "checkpoint"
    model = GPTQModel.load(
        str(MODEL_ID),
        GGUFConfig(bits="q4_k_m", offload_to_disk=False),
        device="cuda:0",
        dtype=torch.float16,
    )
    model.quantize()
    model.save(str(checkpoint))
    del model
    torch.cuda.empty_cache()

    packaged = package(
        GGUFPackageConfig(model_path=str(checkpoint), out_path=str(root / "packaged.gguf"))
    )
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
    prompt = tokenizer.apply_chat_template(
        MESSAGES, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]

    llama = llama_cpp.Llama(
        model_path=str(packaged), n_gpu_layers=-1, n_ctx=256, logits_all=True, verbose=False
    )
    try:
        yield {
            "torch": torch,
            "tokenizer": tokenizer,
            "prompt": prompt,
            "prompt_ids": prompt_ids,
            "llama": llama,
            "packaged": packaged,
            "checkpoint": checkpoint,
        }
    finally:
        llama.close()


def _llama_greedy(llama, prompt_ids, steps):
    llama.reset()
    llama.eval(prompt_ids)
    first = None
    tokens = []
    for _ in range(steps):
        logits = llama.scores[llama.n_tokens - 1].astype("float32").copy()
        if first is None:
            first = logits
        token = int(logits.argmax())
        tokens.append(token)
        llama.eval([token])
    return first, tokens


def _gptqmodel_greedy(torch, model, prompt_ids, steps):
    first = None
    tokens = []
    current = torch.tensor([prompt_ids], device="cuda:0")
    with torch.inference_mode():
        for _ in range(steps):
            logits = model.model(input_ids=current).logits[0, -1].float()
            if first is None:
                first = logits.detach().cpu().numpy()
            token = int(torch.argmax(logits))
            tokens.append(token)
            current = torch.cat(
                [current, torch.tensor([[token]], device="cuda:0")], dim=1
            )
    return first, tokens


def test_chat_template_round_trips_and_renders_identically(engines):
    from llama_cpp.llama_chat_format import Jinja2ChatFormatter

    tokenizer = engines["tokenizer"]
    llama = engines["llama"]
    template = llama.metadata.get("tokenizer.chat_template")

    assert template == tokenizer.chat_template

    formatter = Jinja2ChatFormatter(
        template=template,
        eos_token="<|im_end|>",
        bos_token="<|endoftext|>",
        add_generation_prompt=True,
        stop_token_ids=[151645],
    )
    rendered = formatter(messages=MESSAGES, enable_thinking=False).prompt
    assert rendered == engines["prompt"]


def test_tokenizer_matches(engines):
    llama = engines["llama"]
    prompt = engines["prompt"]
    prompt_ids = engines["prompt_ids"]

    assert llama.tokenize(prompt.encode("utf-8"), add_bos=False, special=True) == prompt_ids


def test_sampling_defaults_are_readable(engines):
    llama = engines["llama"]
    metadata = llama.metadata

    # Qwen3-0.6B's generation_config.json: temperature 0.6, top_k 20, top_p 0.95.
    assert float(metadata["general.sampling.temp"]) == pytest.approx(0.6)
    assert int(metadata["general.sampling.top_k"]) == 20
    assert float(metadata["general.sampling.top_p"]) == pytest.approx(0.95)


def test_greedy_generation_and_logits_match(engines):
    torch = engines["torch"]
    from gptqmodel import GPTQModel

    llama = engines["llama"]
    prompt_ids = engines["prompt_ids"]
    llama_logits, llama_tokens = _llama_greedy(llama, prompt_ids, NEW_TOKENS)

    model = GPTQModel.load(str(engines["checkpoint"]), device="cuda:0", dtype=torch.float16)
    try:
        hf_logits, hf_tokens = _gptqmodel_greedy(torch, model, prompt_ids, NEW_TOKENS)
    finally:
        del model
        torch.cuda.empty_cache()

    flips = sum(a != b for a, b in zip(llama_tokens, hf_tokens))
    assert flips <= MAX_TOKEN_FLIPS, (
        f"llama.cpp {llama_tokens}\nGPT-QModel {hf_tokens}"
    )

    assert int(llama_logits.argmax()) == int(hf_logits.argmax())
    cosine = float(
        (llama_logits @ hf_logits)
        / (float((llama_logits**2).sum()) ** 0.5 * float((hf_logits**2).sum()) ** 0.5)
    )
    assert cosine >= MIN_LOGIT_COSINE, cosine
