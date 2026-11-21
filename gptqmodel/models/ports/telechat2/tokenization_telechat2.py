# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Tele-AI SentencePiece tokenizer, using Transformers 5's Python backend."""

import json
from pathlib import Path
from typing import cast

from transformers import AddedToken
from transformers.tokenization_python import PythonBackend
from transformers.utils.hub import cached_file


class TelechatTokenizerFactory:
    @classmethod
    def from_pretrained(cls, path, **kwargs):
        import json

        options = {key: kwargs[key] for key in ("subfolder", "revision", "cache_dir", "local_files_only", "token") if key in kwargs}
        sentencepiece = cached_file(path, "tokenizer.model", _raise_exceptions_for_missing_entries=False, **options)
        if sentencepiece is not None:
            return Telechat2Tokenizer.from_pretrained(path, **kwargs)
        serialized = cached_file(path, "tokenizer.json", _raise_exceptions_for_missing_entries=False, **options)
        if serialized is None:
            raise RuntimeError("TeleChat requires tokenizer.model or the released serialized BPE tokenizer.json")
        payload = json.loads(Path(serialized).read_text())
        pretokenizer = payload.get("pre_tokenizer", {})
        bytelevel = pretokenizer.get("type") == "ByteLevel" or (
            pretokenizer.get("type") == "Sequence" and any(
                part.get("type") == "ByteLevel" for part in pretokenizer.get("pretokenizers", [])
            )
        )
        if payload.get("model", {}).get("type") != "BPE" or not bytelevel:
            raise ValueError("Unqualified TeleChat serialized tokenizer backend")
        from transformers import TokenizersBackend

        return TokenizersBackend.from_pretrained(path, **kwargs)


class Telechat2Tokenizer(PythonBackend):
    vocab_files_names = {"vocab_file": "tokenizer.model"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, vocab_file, unk_token="<unk>", bos_token="<_start>", eos_token="<_end>",
                 pad_token="<_pad>", sp_model_kwargs=None, add_bos_token=True, add_eos_token=False,
                 clean_up_tokenization_spaces=False, telechat_tokenizer_policy=None, **kwargs):
        import sentencepiece as spm

        self.sp_model_kwargs = dict(sp_model_kwargs or {})
        try:
            self._source_proto = Path(vocab_file).read_bytes()
            self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
            self.sp_model.LoadFromSerializedProto(self._source_proto)
        except (OSError, RuntimeError, TypeError) as exc:
            raise RuntimeError("TeleChat requires a valid tokenizer.model SentencePiece asset") from exc
        self.vocab_file = str(vocab_file)
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        dialect = self.sp_model.IdToPiece(9)
        self.telechat_tokenizer_policy = telechat_tokenizer_policy or (
            "modern" if dialect == "<tool_call>" else "legacy" if dialect == "<_control1>" else None
        )
        if self.telechat_tokenizer_policy not in {"modern", "legacy"}:
            raise ValueError("Unqualified TeleChat SentencePiece vocabulary dialect")
        def special(token):
            return AddedToken(token, lstrip=False, rstrip=False, special=True) if isinstance(token, str) else token
        tokens = {"bos_token": special(bos_token), "eos_token": special(eos_token), "pad_token": special(pad_token)}
        if self.telechat_tokenizer_policy == "legacy":
            tokens["unk_token"] = special(unk_token)
        # This Python backend owns the exact SP token IDs and AddedToken objects.
        decoder = dict(kwargs.pop("added_tokens_decoder", {}))
        for token in tokens.values():
            if isinstance(token, AddedToken):
                index = self.sp_model.PieceToId(str(token))
                if self.sp_model.IdToPiece(index) == str(token):
                    decoder.setdefault(index, AddedToken(token.content, single_word=token.single_word,
                        lstrip=token.lstrip, rstrip=token.rstrip, normalized=token.normalized, special=True))
        super().__init__(**tokens, added_tokens_decoder=decoder,
                         telechat_tokenizer_policy=self.telechat_tokenizer_policy,
                         add_bos_token=add_bos_token, add_eos_token=add_eos_token,
                         sp_model_kwargs=self.sp_model_kwargs,
                         clean_up_tokenization_spaces=clean_up_tokenization_spaces, **kwargs)

    @property
    def vocab_size(self):
        return self.sp_model.GetPieceSize()

    def get_vocab(self):
        return {self.sp_model.IdToPiece(index): index for index in range(self.vocab_size)} | self.added_tokens_encoder

    def _tokenize(self, text, **kwargs):
        return self.sp_model.EncodeAsPieces(text)

    def _convert_token_to_id(self, token):
        return self.sp_model.PieceToId(token)

    def _convert_id_to_token(self, index):
        return self.sp_model.IdToPiece(index)

    def convert_tokens_to_string(self, tokens):
        pieces, output = [], ""
        for token in tokens:
            if token in self.all_special_tokens:
                output += self.sp_model.DecodePieces(pieces) + token
                pieces = []
            else:
                pieces.append(token)
        return output + self.sp_model.DecodePieces(pieces)

    def save_vocabulary(self, save_directory, filename_prefix=None):
        target = Path(save_directory) / ((filename_prefix + "-" if filename_prefix else "") + "tokenizer.model")
        target.write_bytes(self._source_proto)
        return (str(target),)

    def _save_pretrained(self, save_directory, file_names, legacy_format=None, filename_prefix=None):
        files = super()._save_pretrained(save_directory, file_names, legacy_format, filename_prefix)
        path = Path(save_directory) / ((filename_prefix + "-" if filename_prefix else "") + "tokenizer_config.json")
        config = json.loads(path.read_text())
        config.update(add_bos_token=self.add_bos_token, add_eos_token=self.add_eos_token)
        path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
        return files

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        bos = [cast(int, self.bos_token_id)] if self.add_bos_token else []
        eos = [cast(int, self.eos_token_id)] if self.add_eos_token else []
        result = bos + token_ids_0 + eos
        return result if token_ids_1 is None else result + bos + token_ids_1 + eos

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        if already_has_special_tokens:
            return super().get_special_tokens_mask(token_ids_0, token_ids_1, already_has_special_tokens=True)
        bos, eos = [1] if self.add_bos_token else [], [1] if self.add_eos_token else []
        result = bos + [0] * len(token_ids_0) + eos
        return result if token_ids_1 is None else result + bos + [0] * len(token_ids_1) + eos

    def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        first = len(token_ids_0) + int(self.add_bos_token) + int(self.add_eos_token)
        second = 0 if token_ids_1 is None else len(token_ids_1) + int(self.add_bos_token) + int(self.add_eos_token)
        return [0] * first + [1] * second
