# Copyright (c) The InternLM team and The HuggingFace Inc. team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Based on transformers/src/transformers/models/llama/tokenization_llama.py.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from pinned InternLM tokenizer sources; see NOTICE.
"""Checkpoint-native InternLM SentencePiece tokenization on Transformers 5."""

import json
import os
from shutil import copyfile

from transformers.tokenization_python import PreTrainedTokenizer


class InternLMTokenizer(PreTrainedTokenizer):
    vocab_files_names = {"vocab_file": "tokenizer.model"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self, vocab_file, unk_token="<unk>", bos_token="<s>", eos_token="</s>",
        pad_token="</s>", sp_model_kwargs=None, add_bos_token=True,
        add_eos_token=False, decode_with_prefix_space=False,
        clean_up_tokenization_spaces=False, **kwargs,
    ):
        import sentencepiece as spm

        if vocab_file is None:
            raise ValueError(f"{type(self).__name__} requires tokenizer.model")
        self.sp_model_kwargs = {} if sp_model_kwargs is None else dict(sp_model_kwargs)
        self.vocab_file = os.fspath(vocab_file)
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.decode_with_prefix_space = decode_with_prefix_space
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        with open(self.vocab_file, "rb") as file:
            self._vocab_bytes = file.read()
        try:
            self.sp_model.LoadFromSerializedProto(self._vocab_bytes)
        except RuntimeError as error:
            raise RuntimeError(f"{type(self).__name__} could not parse vocabulary: {self.vocab_file}") from error
        super().__init__(
            bos_token=bos_token, eos_token=eos_token, unk_token=unk_token, pad_token=pad_token,
            sp_model_kwargs=self.sp_model_kwargs, add_bos_token=add_bos_token,
            add_eos_token=add_eos_token, decode_with_prefix_space=decode_with_prefix_space,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces, **kwargs,
        )

    @property
    def vocab_size(self):
        return self.sp_model.get_piece_size()

    @property
    def bos_token_id(self):
        return self.sp_model.bos_id()

    @property
    def eos_token_id(self):
        return self.sp_model.eos_id()

    def get_vocab(self):
        vocab = {self.sp_model.IdToPiece(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text):
        return self.sp_model.encode(text, out_type=str)

    def _convert_token_to_id(self, token):
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        return self.sp_model.IdToPiece(index)

    def convert_tokens_to_string(self, tokens):
        current_sub_tokens = []
        out_string = ""
        prev_is_special = False
        for token in tokens:
            if token in self.all_special_tokens:
                if not prev_is_special:
                    out_string += " "
                out_string += self.sp_model.decode(current_sub_tokens) + token
                prev_is_special = True
                current_sub_tokens = []
            else:
                current_sub_tokens.append(token)
                prev_is_special = False
        out_string += self.sp_model.decode(current_sub_tokens)
        # Upstream's prefix-space helper always adds then removes one space for
        # string tokens. Preserve its result, including special-token spacing.
        return self.clean_up_tokenization(out_string)

    def save_vocabulary(self, save_directory, filename_prefix=None):
        if not os.path.isdir(save_directory):
            raise ValueError(f"Vocabulary path must be a directory: {save_directory}")
        name = (filename_prefix + "-" if filename_prefix else "") + self.vocab_files_names["vocab_file"]
        out_vocab_file = os.path.join(save_directory, name)
        if os.path.isfile(self.vocab_file):
            if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
                copyfile(self.vocab_file, out_vocab_file)
        else:
            with open(out_vocab_file, "wb") as file:
                # Newer SentencePiece mutates legacy NUL piece metadata on load.
                file.write(self._vocab_bytes)
        return (out_vocab_file,)

    def _save_pretrained(self, save_directory, file_names, legacy_format=None, filename_prefix=None):
        files = super()._save_pretrained(save_directory, file_names, legacy_format, filename_prefix)
        # HF5 drops these flags for native tokenizer.json post-processors; this
        # SentencePiece implementation still owns their input construction.
        with open(file_names[0], encoding="utf-8") as file:
            config = json.load(file)
        config.update(add_bos_token=self.add_bos_token, add_eos_token=self.add_eos_token)
        with open(file_names[0], "w", encoding="utf-8") as file:
            json.dump(config, file, ensure_ascii=False, indent=2, sort_keys=True)
            file.write("\n")
        return files

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        output = ([self.bos_token_id] if self.add_bos_token else []) + token_ids_0
        if token_ids_1 is not None:
            output += token_ids_1
        return output + ([self.eos_token_id] if self.add_eos_token else [])

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True,
            )
        length = len(token_ids_0) + (len(token_ids_1) if token_ids_1 is not None else 0)
        return ([1] if self.add_bos_token else []) + [0] * length + ([1] if self.add_eos_token else [])

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        return [0] * len(self.build_inputs_with_special_tokens(token_ids_0, token_ids_1))
