# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Runtime helpers for the ggml library that backs the GGUF_CPP kernels.

GPTQModel loads ``libggml-base``/``libggml-cpu``/``libggml-cuda`` from the
installed ``llama-cpp-python`` wheel and binds the public C API with ctypes.
ggml v1.0.0 is the first release that promises a stable public C API, and
``llama-cpp-python>=0.3.37`` is the first wheel that vendors it.

The helpers here only parse what ``ggml_version()`` reports. They never import
``llama-cpp-python``, so importing this module stays cheap and works when the
optional dependency is absent.
"""

from __future__ import annotations

from packaging.version import InvalidVersion, Version


# First ggml release with the stable public C API that GGUF_CPP targets.
GGML_1_0 = Version("1.0.0")

# Shared library that exports ``ggml_version`` in every ggml release.
GGML_VERSION_SYMBOL = "ggml_version"


def parse_ggml_version(raw: str | bytes | None) -> Version | None:
    """Parse the string returned by ``ggml_version()``.

    ``ggml_version()`` returns a semantic version such as ``"1.0.0"``. Older
    builds may report a ``v`` prefix or a non-semantic build tag (for example
    ``"b4567"``); those return ``None`` instead of raising.
    """

    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "replace")
    text = str(raw).strip()
    if not text:
        return None
    if text[:1] in {"v", "V"}:
        text = text[1:]
    try:
        return Version(text)
    except InvalidVersion:
        return None


def ggml_version_at_least(raw: str | bytes | None, minimum: Version = GGML_1_0) -> bool:
    """Return whether a reported ggml version satisfies ``minimum``.

    An unparseable build tag is treated as unknown, not as 1.0, so callers can
    choose whether to warn or fall back.
    """

    version = parse_ggml_version(raw)
    return version is not None and version >= minimum


def is_ggml_1_0_or_newer(raw: str | bytes | None) -> bool:
    return ggml_version_at_least(raw, GGML_1_0)
