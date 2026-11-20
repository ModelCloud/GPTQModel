# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Reproduce the W3.5 Llama 3.2 1B Instruct quality comparison on NVIDIA CUDA."""

import json
import math
import os
import random
import re
from decimal import Decimal
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


_BITS = 3.5
_PROJECTIONS = 112
_CALIBRATION_SEED = 2718
_CALIBRATION_LENGTH = 512
_CALIBRATION_SAMPLES = 256
_WIKI_CONTEXT = 2048
_WIKI_STRIDE = 1024
_GSM_QUESTIONS = 256
_GSM_BATCH = 16
_ANSWER_NUMBER = r"[-+]?(?:[0-9][0-9,]*(?:\.[0-9]+)?|\.[0-9]+)"
_ANSWER_MARKER = re.compile(
    r"(?:final\s+answer(?:\s+is)?\s*[:=]?|####|the\s+answer\s+is)\s*\$?\s*"
    rf"({_ANSWER_NUMBER})",
    re.IGNORECASE,
)


def _local_directory(variable):
    value = os.environ.get(variable)
    if not value:
        pytest.fail(f"Set {variable} as described in docs/exl3-native-decode.md")
    path = Path(value).expanduser()
    assert path.is_dir(), f"{variable} must point to a directory ({path})"
    return path


def _calibration_rows(data_dir, tokenizer):
    from datasets import Dataset, concatenate_datasets

    shards = sorted(data_dir.glob("c4-train-*.arrow"))
    assert shards, f"No C4 train Arrow shards in {data_dir}"
    corpus = concatenate_datasets([Dataset.from_file(str(path)).select_columns("text") for path in shards])
    rng = random.Random(_CALIBRATION_SEED)
    rows = []
    for index in rng.sample(range(len(corpus)), min(8192, len(corpus))):
        text = corpus[index]["text"]
        if len(text) < _CALIBRATION_LENGTH:
            continue
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) < _CALIBRATION_LENGTH:
            continue
        start = rng.randrange(len(tokens) - _CALIBRATION_LENGTH + 1)
        rows.append({
            "input_ids": tokens[start:start + _CALIBRATION_LENGTH],
            "attention_mask": [1] * _CALIBRATION_LENGTH,
        })
        if len(rows) == _CALIBRATION_SAMPLES:
            return rows
    pytest.fail(f"Only found {len(rows)} full-length calibration samples in {data_dir}")


def _assert_saved_trellises(path):
    from safetensors import safe_open

    trellises = {}
    for shard in path.glob("*.safetensors"):
        with safe_open(str(shard), framework="pt", device="cpu") as tensors:
            for name in tensors.keys():
                if name.endswith(".trellis"):
                    trellis = tensors.get_slice(name)
                    trellises[name] = (trellis.get_dtype(), trellis.get_shape()[-1])
    assert len(trellises) == _PROJECTIONS
    assert set(trellises.values()) == {("I16", int(16 * _BITS))}


def _summed_nll(logits, labels):
    nll = F.cross_entropy(logits.flatten(0, 1).float(), labels.flatten(), reduction="sum").item()
    assert math.isfinite(nll)
    return nll


@torch.inference_mode()
def _wikitext_perplexity(model, tokens, device):
    previous_end = total_nll = scored = 0
    for start in range(0, len(tokens), _WIKI_STRIDE):
        end = min(start + _WIKI_CONTEXT, len(tokens))
        targets = end - max(start + 1, previous_end)
        ids = torch.tensor([tokens[start:end]], dtype=torch.long, device=device)
        # Overlap windows for context and score each next-token target only once.
        logits = model(input_ids=ids, use_cache=False, logits_to_keep=targets + 1).logits
        assert logits.shape[:2] == (1, targets + 1)
        total_nll += _summed_nll(logits[:, :-1], ids[:, -targets:])
        scored += targets
        previous_end = end
        if end == len(tokens):
            break
    assert scored == len(tokens) - 1
    return math.exp(total_nll / scored)


def _gsm_prompts(tokenizer, questions):
    prompts = [
        tokenizer.apply_chat_template(
            [{
                "role": "user",
                "content": row["question"] + "\n\nShow your work, then write Final answer: followed by the number.",
            }],
            tokenize=False,
            add_generation_prompt=True,
        )
        for row in questions
    ]
    return tokenizer(prompts, add_special_tokens=False)["input_ids"]


def _answer_number(text):
    numbers = _ANSWER_MARKER.findall(text) or re.findall(_ANSWER_NUMBER, text)
    return Decimal(numbers[-1].replace(",", "")) if numbers else None


@torch.inference_mode()
def _gsm_correct(model, tokenizer, prompts, answers, eos_ids, device):
    from transformers import GenerationConfig

    tokenizer.padding_side = "left"
    config = GenerationConfig(
        do_sample=False,
        max_new_tokens=256,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=eos_ids,
    )
    correct = 0
    for start in range(0, len(prompts), _GSM_BATCH):
        end = start + _GSM_BATCH
        inputs = tokenizer.pad({"input_ids": prompts[start:end]}, return_tensors="pt").to(device)
        generated = model.generate(**inputs, generation_config=config)[:, inputs.input_ids.shape[1]:].cpu().tolist()
        for ids, answer in zip(generated, answers[start:end], strict=True):
            stop = next((index for index, token in enumerate(ids) if token in eos_ids), len(ids))
            predicted = _answer_number(tokenizer.decode(ids[:stop], skip_special_tokens=True))
            correct += predicted == answer
    return correct


@torch.inference_mode()
def _cached_wikitext_probe(model, tokens, device):
    context = 256
    starts = sorted(random.Random(1049).sample(range(0, len(tokens) - context - 2, context), 8))
    total_nll = 0.0
    decode_logits = []
    for start in starts:
        ids = torch.tensor([tokens[start:start + context + 1]], dtype=torch.long, device=device)
        prefill = model(input_ids=ids[:, :-1], use_cache=True)
        total_nll += _summed_nll(prefill.logits, ids[:, 1:])
        step = model(input_ids=ids[:, -1:], past_key_values=prefill.past_key_values, use_cache=True)
        decode_logits.append(step.logits[0, 0].float().cpu())
    return math.exp(total_nll / (len(starts) * context)), torch.stack(decode_logits)


@pytest.mark.cuda
@pytest.mark.integration
@pytest.mark.model
@pytest.mark.slow
@pytest.mark.exl3_cuda_e2e
def test_llama3_2_1b_instruct_exl3_w35_quantize_reload_and_quality(tmp_path):
    if not torch.cuda.is_available() or torch.version.hip is not None:
        pytest.skip("requires NVIDIA CUDA")
    device = torch.device("cuda", torch.cuda.current_device())
    if torch.cuda.get_device_capability(device)[0] < 8:
        pytest.skip("requires NVIDIA Ampere or newer")

    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    from gptqmodel import BACKEND, CalibrationConfig, GPTQModel
    from gptqmodel.nn_modules.exllamav3 import ExllamaV3Linear
    from gptqmodel.nn_modules.exllamav3_torch import ExllamaV3TorchLinear
    from gptqmodel.quantization.config import EXL3Config

    source = _local_directory("GPTQMODEL_EXL3_E2E_MODEL_DIR")
    data_dir = _local_directory("GPTQMODEL_EXL3_E2E_DATA_DIR")
    checkpoint = tmp_path / "exl3-llama-1b-w35"
    options = {"dtype": torch.bfloat16, "attn_implementation": "sdpa"}

    tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    eos_ids = GenerationConfig.from_pretrained(source, local_files_only=True).eos_token_id
    wiki_text = "\n\n".join(Dataset.from_file(str(data_dir / "wikitext-test.arrow"))["text"])
    wiki_tokens = tokenizer.encode(wiki_text, add_special_tokens=False)
    assert len(wiki_tokens) == 289_076
    examples = [json.loads(line) for line in (data_dir / "gsm8k-test.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(examples) == 1319
    indices = sorted(random.Random(1618).sample(range(len(examples)), _GSM_QUESTIONS))
    questions = [examples[index] for index in indices]
    prompts = _gsm_prompts(tokenizer, questions)
    answers = [Decimal(row["answer"].rsplit("####", 1)[1].strip().replace(",", "")) for row in questions]

    dense = AutoModelForCausalLM.from_pretrained(source, device_map={"": device}, local_files_only=True, **options)
    base_ppl = _wikitext_perplexity(dense.eval(), wiki_tokens, device)
    base_score = _gsm_correct(dense, tokenizer, prompts, answers, eos_ids, device)
    del dense
    print(f"BF16 WikiText-2 PPL {base_ppl:.4f}, GSM8K {base_score}/{_GSM_QUESTIONS}", flush=True)

    torch.manual_seed(_CALIBRATION_SEED)
    config = EXL3Config(bits=_BITS, codebook="mul1", offload_to_disk=False)
    quantized = GPTQModel.load(str(source), quantize_config=config, device=str(device), dtype=torch.bfloat16)
    quantized.quantize(CalibrationConfig(dataset=_calibration_rows(data_dir, tokenizer), batch_size=8))
    quantized.save(str(checkpoint))
    del quantized
    _assert_saved_trellises(checkpoint)

    reloaded = GPTQModel.load(str(checkpoint), device=str(device), backend=BACKEND.EXL3_EXLLAMA_V3, **options)
    assert reloaded.quantize_config.runtime_bits == _BITS
    assert reloaded.quantize_config.codebook == "mul1"
    native = reloaded.model.eval()
    projections = [module for module in native.modules() if isinstance(module, ExllamaV3Linear)]
    assert len(projections) == _PROJECTIONS
    for projection in projections:
        assert projection.trellis.device == device
        assert projection.trellis.dtype == torch.int16
        assert projection.trellis.shape[-1] == int(16 * _BITS)
    del projections, projection

    saved_tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    saved_tokenizer.pad_token = tokenizer.pad_token
    assert saved_tokenizer.pad_token_id == tokenizer.pad_token_id
    assert saved_tokenizer.encode(wiki_text, add_special_tokens=False) == wiki_tokens
    assert _gsm_prompts(saved_tokenizer, questions) == prompts
    native_ppl = _wikitext_perplexity(native, wiki_tokens, device)
    native_score = _gsm_correct(native, saved_tokenizer, prompts, answers, eos_ids, device)
    native_probe_ppl, native_decode = _cached_wikitext_probe(native, wiki_tokens, device)
    del reloaded, native
    print(f"EXL3 W3.5 WikiText-2 PPL {native_ppl:.4f}, GSM8K {native_score}/{_GSM_QUESTIONS}", flush=True)

    reference = GPTQModel.load(str(checkpoint), device=str(device), backend=BACKEND.EXL3_TORCH, **options).model.eval()
    assert sum(isinstance(module, ExllamaV3TorchLinear) for module in reference.modules()) == _PROJECTIONS
    reference_ppl, reference_decode = _cached_wikitext_probe(reference, wiki_tokens, device)
    print(f"Packed-weight WikiText-2 probe PPL {native_probe_ppl:.4f} CUDA, {reference_ppl:.4f} Torch", flush=True)

    # The A100 run measured +5.59% PPL and -9/256 correct answers. Bound
    # the quality change against the dense model evaluated in this run.
    assert native_ppl <= base_ppl * 1.10
    assert (base_score - native_score) / _GSM_QUESTIONS <= 0.10
    # Identical packed weights isolate kernel drift from quantization error.
    # A100 measured 0.071% probe PPL drift and 0.21875 max cached-logit drift.
    assert native_probe_ppl == pytest.approx(reference_ppl, rel=0.01)
    torch.testing.assert_close(native_decode, reference_decode, rtol=0, atol=0.5)
