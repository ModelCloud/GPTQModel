# Copyright (c) OpenMOSS and the HuggingFace teams.
# SPDX-License-Identifier: Apache-2.0
"""MOSS's GPT-2 byte BPE with its source BOS/pair semantics.

Class selection is package-owned; Tokenicer owns corrective normalization.
"""

import regex as re
from tokenizers.processors import TemplateProcessing

from ..native.gpt2.tokenization_gpt2 import GPT2Tokenizer


class MossTokenizer(GPT2Tokenizer):
    def __init__(self, *args, add_bos_token=None, eos_token="<eom>", **kwargs):
        # A native tokenizer.json owns its post-processor. Do not overwrite a
        # serialized BOS template with the source default when reloading it.
        if add_bos_token is not None:
            kwargs["add_bos_token"] = add_bos_token
        super().__init__(*args, eos_token=eos_token, **kwargs)
        if add_bos_token is True:
            self.backend_tokenizer.post_processor = TemplateProcessing(
                single=f"{self.bos_token} $A", pair=f"{self.bos_token} $A {self.bos_token} $B",
                special_tokens=[(str(self.bos_token), self.bos_token_id)],
            )
        if add_bos_token is None:
            inserted = self.backend_tokenizer.encode("", add_special_tokens=True).ids
            object.__setattr__(self, "_add_bos_token", inserted == [self.bos_token_id])

    def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=None,
               truncate_before_pattern=None, **kwargs):
        text = super().decode(token_ids, skip_special_tokens=skip_special_tokens,
                              clean_up_tokenization_spaces=clean_up_tokenization_spaces, **kwargs)
        if truncate_before_pattern:
            text = self.truncate(text, truncate_before_pattern)
        return text

    @staticmethod
    def truncate(completion, truncate_before_pattern):
        # Preserve the source's opt-in code-completion stopping semantics.
        for keyword in ("print", "def"):
            repeated = list(re.finditer("^" + keyword, completion, re.MULTILINE))
            if len(repeated) > 1:
                completion = completion[:repeated[1].start()]
        positions = [match.start() for pattern in truncate_before_pattern
                     if (match := re.search(pattern, completion, re.MULTILINE)) is not None]
        return completion[:min(positions)] if positions else completion
