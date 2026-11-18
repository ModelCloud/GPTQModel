# SPDX-FileCopyrightText: 2018 The Open AI Team Authors and The HuggingFace Inc. team.
# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""PanGu-α jieba/SentencePiece adapter for Transformers 5; see NOTICE."""

from pathlib import Path

from transformers.tokenization_python import PreTrainedTokenizer


class GPTPanguTokenizer(PreTrainedTokenizer):
    vocab_files_names = {"vocab_file": "vocab.model"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, vocab_file, eos_token="<eot>", **kwargs):
        try:
            import jieba
        except ImportError as exc:
            raise ImportError(
                "The PanGu-α tokenizer requires jieba. Install it with pip install 'gptqmodel[pangu-alpha]'."
            ) from exc

        import sentencepiece as spm

        if vocab_file is None:
            raise RuntimeError("PanGu-α tokenizer requires vocab.model")
        self.vocab_file = str(vocab_file)
        try:
            self._vocab_bytes = Path(vocab_file).read_bytes()
            self.sp = spm.SentencePieceProcessor(model_proto=self._vocab_bytes)
        except (OSError, RuntimeError, ValueError) as exc:
            raise RuntimeError(f"Cannot load PanGu-α vocab.model: {vocab_file}") from exc
        if not self.sp.get_piece_size():
            raise RuntimeError(f"PanGu-α vocab.model has no pieces: {vocab_file}")
        self._segmenter = jieba.Tokenizer()
        self._translator = str.maketrans(" \n", "\u2582\u2583")
        super().__init__(eos_token=eos_token, **kwargs)

    @property
    def vocab_size(self):
        return self.sp.get_piece_size()

    def get_vocab(self):
        return {**{self.sp.id_to_piece(i): i for i in range(self.vocab_size)}, **self.added_tokens_encoder}

    def _tokenize(self, text, **kwargs):
        segmented = " ".join(piece.translate(self._translator) for piece in self._segmenter.cut(text, cut_all=False))
        return self.sp.encode(segmented, out_type=str)

    def tokenize(self, text, **kwargs):
        return self._tokenize(text, **kwargs)

    def _convert_token_to_id(self, token):
        return self.sp.piece_to_id(token)

    def _convert_id_to_token(self, index):
        return self.sp.id_to_piece(index)

    def convert_tokens_to_string(self, tokens):
        return self.sp.decode(tokens).replace(" ", "").replace("\u2582", " ").replace("\u2583", "\n")

    def save_vocabulary(self, save_directory, filename_prefix=None):
        path = Path(save_directory) / ((filename_prefix + "-" if filename_prefix else "") + "vocab.model")
        path.write_bytes(self._vocab_bytes)
        return (str(path),)
