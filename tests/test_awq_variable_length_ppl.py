# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""AWQ must beat RTN when calibration batches have different sequence lengths.

Issue #4063 item 1: variable-length captures could not be concatenated, so
every dense module kept only the latest capture. Calibration is sorted by
length descending, so that was the shortest sample and AWQ ended up no better
than RTN. This test quantizes the same real model with variable-length C4
documents, reloads both checkpoints (the in-memory checkpoint is a separate
issue), and measures wikitext-2 perplexity.
"""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from gptqmodel import GPTQModel
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.quantization.config import QuantizeConfig, RTNConfig


pytestmark = [pytest.mark.gpu, pytest.mark.slow]

MODEL_PATH = Path(
    os.environ.get(
        "GPTQMODEL_AWQ_PPL_TEST_MODEL",
        "/monster/data/model/Qwen2.5-0.5B-Instruct",
    )
)
C4_LOCAL = Path("/monster/data/model/dataset/c4-train.00000-of-01024.json.gz")
WIKITEXT_TEXT_ENV = "GPTQMODEL_AWQ_PPL_WIKITEXT"
CALIBRATION_DOCS = 128
PPL_CHUNKS = 80
SEQ_LEN = 512


def _calibration_texts(count: int) -> list[str]:
    if C4_LOCAL.is_file():
        texts: list[str] = []
        with gzip.open(C4_LOCAL, "rt", encoding="utf-8") as handle:
            for line in handle:
                if len(texts) >= count:
                    break
                text = json.loads(line).get("text", "")
                if text and text.strip():
                    texts.append(text)
        return texts

    datasets = pytest.importorskip("datasets", reason="C4 calibration needs `datasets`")
    split = datasets.load_dataset(
        "allenai/c4",
        data_files="en/c4-train.00000-of-01024.json.gz",
        split="train",
    ).select(range(count))
    return [row for row in split["text"] if row and row.strip()]


def _wikitext_text() -> str:
    local_text = os.environ.get(WIKITEXT_TEXT_ENV)
    if local_text:
        path = Path(local_text)
        if not path.is_file():
            pytest.skip(f"{WIKITEXT_TEXT_ENV} does not point to a file: {path}")
        return path.read_text(encoding="utf-8")

    datasets = pytest.importorskip("datasets", reason="wikitext-2 needs `datasets`")
    split = datasets.load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        split="test",
    )
    return "\n\n".join(split["text"])


def _ppl_chunks(tokenizer) -> torch.Tensor:
    ids = tokenizer(
        _wikitext_text(),
        return_tensors="pt",
        add_special_tokens=False,
    )["input_ids"][0]
    usable = (ids.numel() // SEQ_LEN) * SEQ_LEN
    chunks = ids[:usable].view(-1, SEQ_LEN)
    if chunks.shape[0] < PPL_CHUNKS:
        pytest.skip(f"wikitext fixture only produced {chunks.shape[0]} chunks")
    return chunks[:PPL_CHUNKS]


@torch.inference_mode()
def _perplexity(model, chunks: torch.Tensor, device: torch.device) -> float:
    total_nll = 0.0
    total_tokens = 0
    for row in chunks:
        input_ids = row.unsqueeze(0).to(device)
        out = model(input_ids=input_ids, labels=input_ids, use_cache=False)
        predicted = input_ids.shape[1] - 1
        total_nll += float(out.loss) * predicted
        total_tokens += predicted
    return float(torch.exp(torch.tensor(total_nll / total_tokens)))


def _quantize_and_measure(model_path, quantize_config, calibration, chunks, tmp_path, label) -> float:
    device = torch.device("cuda:0")
    model = GPTQModel.load(model_path, quantize_config=quantize_config, dtype=torch.float16)
    model.quantize(calibration, batch_size=1, calibration_sort="desc")

    save_path = tmp_path / label
    model.save(str(save_path))
    del model
    torch.cuda.empty_cache()

    reloaded = GPTQModel.load(str(save_path), device="cuda:0", dtype=torch.float16)
    ppl = _perplexity(reloaded.model, chunks, device)
    del reloaded
    torch.cuda.empty_cache()
    return ppl


def test_awq_beats_rtn_with_variable_length_calibration(tmp_path):
    if not MODEL_PATH.is_dir():
        pytest.skip(f"local model fixture is unavailable: {MODEL_PATH}")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the AWQ/RTN perplexity comparison")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    calibration = _calibration_texts(CALIBRATION_DOCS)
    lengths = {len(tokenizer(text, add_special_tokens=False)["input_ids"]) for text in calibration}
    assert len(lengths) > 1, "calibration must contain variable-length documents"
    chunks = _ppl_chunks(tokenizer)

    common = {
        "bits": 4,
        "group_size": 128,
        "sym": False,
        "format": FORMAT.GEMM,
        "device": "cuda:0",
        "calibration_data_device": "cuda:0",
    }
    rtn_ppl = _quantize_and_measure(
        str(MODEL_PATH), RTNConfig(**common), calibration, chunks, tmp_path, "rtn"
    )
    awq_ppl = _quantize_and_measure(
        str(MODEL_PATH),
        QuantizeConfig(method=METHOD.AWQ, **common),
        calibration,
        chunks,
        tmp_path,
        "awq",
    )

    print({"rtn_ppl": rtn_ppl, "awq_ppl": awq_ppl, "calibration_lengths": sorted(lengths)})
    # Require a real improvement, not a tie inside run-to-run noise.
    assert awq_ppl < rtn_ppl * 0.995, (
        f"AWQ perplexity {awq_ppl:.4f} must be at least 0.5% below RTN "
        f"{rtn_ppl:.4f}; variable-length calibration is still collapsing to "
        "one sample"
    )
