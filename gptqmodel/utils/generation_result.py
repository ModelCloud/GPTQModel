# SPDX-FileCopyrightText: 2024-2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Shared result type for runtime generation adapters.

Kept free of optional dependencies so an adapter can import it without pulling
in the HTTP server stack.
"""


class GenerationResult(str):
    """Decoded text that also carries how the sequence ended.

    It subclasses ``str`` so every existing consumer keeps working unchanged:
    adapters that already returned text keep returning text, and callers that
    compare, concatenate, or print the value are unaffected. Adapters that know
    the real token count and termination reason attach them here, which lets
    the OpenAI server report actual usage instead of a fabricated
    ``completion_tokens=0`` and a constant ``finish_reason="stop"``.

    ``completion_tokens`` and ``finish_reason`` are ``None`` when the runtime
    did not report them; that means "unknown", not "zero".
    """

    __slots__ = ("completion_tokens", "finish_reason")

    def __new__(cls, text: str, completion_tokens=None, finish_reason=None):
        if not isinstance(text, str):
            raise TypeError(f"GenerationResult text must be a string, got {type(text).__name__}.")
        instance = super().__new__(cls, text)
        instance.completion_tokens = completion_tokens
        instance.finish_reason = finish_reason
        return instance

    @property
    def text(self) -> str:
        return str(self)

    def __repr__(self) -> str:
        return (
            f"GenerationResult(text={str(self)!r}, completion_tokens={self.completion_tokens!r}, "
            f"finish_reason={self.finish_reason!r})"
        )
