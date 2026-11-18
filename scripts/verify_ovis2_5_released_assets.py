# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Offline verification of the pinned release's real tokenizer and media assets.

Run with an asset directory containing the files in released_tokenizer_evidence.json.
This qualification utility does not download assets or execute checkpoint Python.
"""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from tokenizers import AddedToken, Tokenizer

from gptqmodel.models.ports.ovis2_5.configuration_ovis2_5 import Ovis2_5_Config
from gptqmodel.models.ports.ovis2_5.processing_ovis2_5 import Ovis2_5Processor
from gptqmodel.utils.hf import load_hf_tokenizer


def verify(assets):
    assets = Path(assets)
    port = Path(__file__).resolve().parents[1] / "gptqmodel/models/ports/ovis2_5"
    reference = json.loads((port / "released_tokenizer_evidence.json").read_text())
    for filename, sha256 in reference["assets"].items():
        assert hashlib.sha256((assets / filename).read_bytes()).hexdigest() == sha256, filename
    config = Ovis2_5_Config(**json.loads((port / "release_2B_config.json").read_text()))
    tokenizer = load_hf_tokenizer(str(assets), config=config, trust_remote_code=False, local_files_only=True)
    # Independent Rust BPE oracle plus the source JSON's additional-special-token metadata.
    # In this release <image>/<video> are added by metadata, absent from tokenizer.json.
    raw = Tokenizer.from_file(str(assets / "tokenizer.json"))
    metadata = json.loads((assets / "tokenizer_config.json").read_text())
    for index, entry in sorted(metadata["added_tokens_decoder"].items(), key=lambda pair: int(pair[0])):
        raw.add_special_tokens([AddedToken(**entry)])
        assert raw.token_to_id(entry["content"]) == int(index)
    raw.add_special_tokens([AddedToken(item, normalized=False, special=True)
                            for item in metadata["additional_special_tokens"]])
    processor = Ovis2_5Processor.from_pretrained(str(assets), config=config, tokenizer=tokenizer, local_files_only=True)
    image = Image.fromarray(np.arange(192, dtype=np.uint8).reshape(8, 8, 3))
    for case in reference["evidence"]:
        if "text" in case:
            ids = raw.encode(case["text"], add_special_tokens=False).ids
            assert ids == tokenizer.encode(case["text"], add_special_tokens=False) == case["ids"]
            assert raw.decode(ids, skip_special_tokens=False) == case["decode"]
            assert tokenizer.decode(ids, skip_special_tokens=False) == case["decode"]
        else:
            messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                                      {"type": "text", "text": "What is here? 世界"}]}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                   enable_thinking=case["thinking"])
            assert text == case["chat"]
            before, after = text.split("<image>")
            ids = (raw.encode(before, add_special_tokens=False).ids + [-301] + [-300] * 4 + [-302]
                   + raw.encode(after, add_special_tokens=False).ids)
            actual_ids, pixels, grids = processor.preprocess_inputs(
                messages, min_pixels=4096, max_pixels=4096, enable_thinking=case["thinking"])
            assert ids == actual_ids[0].tolist() == case["ids"]
            assert list(pixels.shape) == case["pixels_shape"]
            assert grids.tolist() == case["grid"]
    with tempfile.TemporaryDirectory() as saved:
        processor.save_pretrained(saved)
        restored = Ovis2_5Processor.from_pretrained(saved, local_files_only=True)
        for case in reference["evidence"]:
            if "text" in case:
                assert restored.tokenizer.encode(case["text"], add_special_tokens=False) == case["ids"]
    print(f"{len(reference['evidence'])} real release token/chat/media cases and processor reload passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assets", type=Path)
    verify(parser.parse_args().assets)
