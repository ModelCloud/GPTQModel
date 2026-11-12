# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Deterministic GSM8K evaluation for native-format Tenstorrent checkpoints.

Five-shot prompting, greedy decoding, and strict answer extraction from the
``#### N`` marker, so a score is reproducible rather than sampled. The metric is
exact match on the final numeric answer -- the standard GSM8K metric.

Usage::

    python scripts/gsm8k_eval.py <model-id-or-path> [limit] [--native CHECKPOINT]

``--chat-template`` wraps each prompt as a single user turn. Only instruction
tuned tokenizers have a template; base models (for example the non-Instruct
Llama-3.2-1B) do not, and the run reports that rather than silently ignoring it.

``--native`` loads a ``tt-blockfloat`` checkpoint produced by
:func:`gptqmodel.utils.tt_blockfloat_quant.save_blockfloat_checkpoint`, so the
same harness scores the dense and the quantized model on identical prompts.

This module is importable: :func:`build_prompt`, :func:`extract`, :func:`gold`
and :func:`run_eval` are used directly by the test suite.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time


NUM = re.compile(r"-?\$?\d[\d,]*\.?\d*")
MARKER = re.compile(r"####\s*(-?\$?\d[\d,]*\.?\d*)")


def load_rows(path: str | None = None) -> list:
    """Read the GSM8K test split as JSONL.

    Defaults to ``$GSM8K_TEST_JSONL`` or ``gsm8k_test.jsonl`` beside the CWD.
    """

    path = path or os.environ.get("GSM8K_TEST_JSONL", "gsm8k_test.jsonl")
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def gold(answer: str) -> str:
    """Extract the reference numeric answer from a GSM8K answer field."""

    return answer.split("####")[-1].strip().replace(",", "")


def build_prompt(rows: list, index: int, tok) -> str:
    """Build a five-shot prompt whose last line asks the question under test.

    Deterministic: the shots are always the five rows immediately preceding
    ``index``, so every run scores the same prompt.
    """

    parts = []
    for j in range(max(0, index - 5), index):
        parts.append(f"Question: {rows[j]['question']}\nAnswer: {rows[j]['answer']}\n\n")
    parts.append(f"Question: {rows[index]['question']}\nAnswer:")
    return "".join(parts)


def apply_chat_template_to_prompt(tok, prompt: str, *, enabled: bool) -> tuple[str, bool]:
    """Wrap a prompt as a single user turn when the tokenizer has a template.

    Returns ``(rendered, applied)``. The ``applied`` flag matters: the existing
    harness in ``tests/eval.py`` silently returns the raw prompt when a tokenizer
    has no template, which makes a mis-specified run look identical to a
    correctly templated one. This reports the truth so a caller can assert it.
    """

    if not enabled:
        return prompt, False
    template = getattr(tok, "chat_template", None)
    if not template or not hasattr(tok, "apply_chat_template"):
        return prompt, False
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(rendered, str) or not rendered.strip():
        return prompt, False
    return rendered, True


def extract(text: str) -> str | None:
    """Pull the predicted numeric answer out of generated text."""

    match = MARKER.search(text)
    if match:
        return match.group(1).replace(",", "").replace("$", "").rstrip(".")
    found = NUM.findall(text)
    if not found:
        return None
    return found[-1].replace(",", "").replace("$", "").rstrip(".")


def run_eval(
    model_id: str,
    limit: int = 200,
    *,
    native: str | None = None,
    max_new_tokens: int = 256,
    rows_path: str | None = None,
    verbose: bool = True,
    chat_template: bool = False,
) -> dict:
    """Score ``model_id`` on GSM8K and return ``{"accuracy", "n", "correct", ...}``."""

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.environ.setdefault("USE_TORCH_XLA", "0")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)

    restored = 0
    if native:
        from gptqmodel.utils.tt_blockfloat_quant import load_native_checkpoint_into_dense_model

        restored = len(load_native_checkpoint_into_dense_model(model, native))
        if verbose:
            print(f"[E] loaded {restored} layers from {native}", flush=True)
    model.eval()

    # A chat template only makes sense for an instruction-tuned tokenizer; base
    # models have none. Record whether it was actually applied so a run cannot
    # silently claim to be templated when it is not.
    templated = bool(getattr(tok, "chat_template", None))
    if chat_template and not templated:
        print(
            f"[E] WARNING: --chat-template requested but {model_id} has no chat "
            "template; prompts are sent untemplated.",
            flush=True,
        )

    rows = load_rows(rows_path)
    n = min(limit, len(rows))
    correct = 0
    started = time.time()
    with torch.no_grad():
        for i in range(n):
            prompt = build_prompt(rows, i, tok)
            prompt, _applied = apply_chat_template_to_prompt(
                tok, prompt, enabled=chat_template
            )
            ids = tok(
                prompt, return_tensors="pt", truncation=True, max_length=2048,
            ).input_ids
            out = model.generate(
                ids, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
            text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
            predicted = extract(text)
            if predicted is not None and predicted == gold(rows[i]["answer"]):
                correct += 1
            if verbose and (i + 1) % 25 == 0:
                print(
                    f"[E] {i+1}/{n} acc={correct/(i+1):.3f} "
                    f"({time.time()-started:.0f}s)",
                    flush=True,
                )

    accuracy = correct / n if n else 0.0
    result = {
        "model": model_id,
        "native": native,
        "restored_layers": restored,
        "n": n,
        "correct": correct,
        "accuracy": accuracy,
        "seconds": round(time.time() - started, 1),
        "chat_template": chat_template,
        "tokenizer_has_template": templated,
    }
    if verbose:
        print(f"[E] RESULT {result}", flush=True)
    return result


def main(argv: list) -> int:
    if not argv:
        print(__doc__)
        return 2
    model_id = argv[0]
    limit = 200
    native = None
    rest = argv[1:]
    if rest and not rest[0].startswith("--"):
        limit = int(rest[0])
    if "--native" in argv:
        native = argv[argv.index("--native") + 1]
    run_eval(model_id, limit, native=native, chat_template="--chat-template" in argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
