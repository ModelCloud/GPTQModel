# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""InternVL-chat calibration must keep the final assistant answer (#4227).

``_extract_history_and_question`` used to drop the answer of a conversation that
ends with an assistant turn, so every caption was removed from the calibration
text before the Hessian was accumulated.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from gptqmodel.models.definitions.internvl_chat import InternVLChatQModel


CAPTION = "A long detailed caption about a dog on a beach."

CAPTION_CONVERSATION = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "x.png"},
            {"type": "text", "text": "Describe this image."},
        ],
    },
    {"role": "assistant", "content": CAPTION},
]


class _FakeConvTemplate:
    """Minimal stand-in for InternVL's ``conv_template``."""

    roles = ("<|im_start|>user\n", "<|im_start|>assistant\n")
    sep = "<|im_end|>"

    def __init__(self):
        self.system_message = ""
        self.messages = []

    def copy(self):
        clone = _FakeConvTemplate()
        clone.system_message = self.system_message
        clone.messages = list(self.messages)
        return clone

    def append_message(self, role, message):
        self.messages.append((role, message))

    def get_prompt(self):
        prompt = self.system_message + "\n" if self.system_message else ""
        for role, message in self.messages:
            if message:
                prompt += role + message + self.sep
            else:
                prompt += role
        return prompt


class _FakeTokenizer:
    def __init__(self):
        self.last_text = None

    def convert_tokens_to_ids(self, token):
        return {"<|im_end|>": 7, "<IMG_CONTEXT>": 5}.get(token, 0)

    def __call__(self, text, return_tensors=None):
        self.last_text = text
        return {
            "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
        }


def _make_model():
    model = object.__new__(InternVLChatQModel)
    nn.Module.__init__(model)
    model.model = SimpleNamespace(
        conv_template=_FakeConvTemplate(),
        system_message="system",
        num_image_token=2,
        config=SimpleNamespace(
            force_image_size=448,
            min_dynamic_patch=1,
            max_dynamic_patch=1,
            use_thumbnail=False,
            vision_config=SimpleNamespace(image_size=448),
        ),
    )
    model.tokenizer = _FakeTokenizer()
    return model


def test_extract_keeps_final_assistant_answer():
    history, question, answer = InternVLChatQModel._extract_history_and_question(
        CAPTION_CONVERSATION
    )

    assert history is None
    assert question == "<image>\nDescribe this image."
    assert answer == CAPTION


def test_extract_multi_turn_keeps_final_answer():
    conversation = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]

    history, question, answer = InternVLChatQModel._extract_history_and_question(conversation)

    assert history == [("q1", "a1")]
    assert question == "q2"
    assert answer == "a2"


def test_extract_without_final_answer_returns_none():
    conversation = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ]

    history, question, answer = InternVLChatQModel._extract_history_and_question(conversation)

    assert history == [("q1", "a1")]
    assert question == "q2"
    assert answer is None


def test_extract_requires_a_user_prompt():
    with pytest.raises(ValueError, match="at least one user prompt"):
        InternVLChatQModel._extract_history_and_question(
            [{"role": "assistant", "content": "orphan answer"}]
        )


def test_build_query_renders_answer_after_question():
    model = _make_model()
    history, question, answer = InternVLChatQModel._extract_history_and_question(
        CAPTION_CONVERSATION
    )

    query, eos_token_id, img_context_token_id = model._build_query(
        question=question,
        history=history,
        num_patches_list=[1],
        answer=answer,
    )

    assert answer in query
    assert query.index("Describe this image.") < query.index(answer)
    assert "<img>" + "<IMG_CONTEXT>" * 2 + "</img>" in query
    assert eos_token_id == 7
    assert img_context_token_id == 5


def test_build_query_without_answer_keeps_generation_prompt():
    model = _make_model()
    history, question, answer = InternVLChatQModel._extract_history_and_question(
        CAPTION_CONVERSATION[:1]
    )

    query, _, _ = model._build_query(
        question=question,
        history=history,
        num_patches_list=[1],
        answer=answer,
    )

    assert answer is None
    assert query.endswith("<|im_start|>assistant\n")


def test_prepare_inputs_for_conversation_includes_caption(monkeypatch):
    model = _make_model()
    monkeypatch.setattr(
        InternVLChatQModel,
        "process_vision_info",
        classmethod(lambda cls, *args, **kwargs: (None, None)),
    )

    tokenized = model.prepare_inputs_for_conversation(CAPTION_CONVERSATION)

    assert CAPTION in model.tokenizer.last_text
    assert model.tokenizer.last_text.index("Describe this image.") < model.tokenizer.last_text.index(CAPTION)
    assert tokenized["eos_token_id"] == 7
    assert tokenized["img_context_token_id"] == 5
