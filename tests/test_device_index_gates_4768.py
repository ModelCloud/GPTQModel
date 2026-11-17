# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Guard the #4768 files against fixed-device-index gates.

``tests/test_hessian.py`` used to skip all 17 tests behind a module-level
``device_count() <= 6`` gate, and four more files gated on ``cuda:6``/``cuda:7``
/``cuda:8``. Those gates skip on every host with fewer GPUs than the lab box
that wrote them, so the tests never run in CI.

AGENTS.md requires capability-relative device selection: use ``cuda:0`` /
``torch.cuda.current_device()`` and skip only when the device count that the
test actually needs is absent. This test fails if a fixed index above 0 comes
back to any of these files.
"""

import re
from pathlib import Path

import pytest


TESTS_DIR = Path(__file__).resolve().parent

AFFECTED_FILES = (
    "test_gptq.py",
    "test_g2h_tensor_quality.py",
    "test_cuda_event_stream_activation_buffer.py",
    "test_fp4_llama3_fp4.py",
)

FORBIDDEN_PATTERNS = (
    (re.compile(r"device_count\(\)\s*<=\s*[1-9]"), "device_count() <= N with N > 0"),
    (re.compile(r"device_count\(\)\s*<\s*[2-9]"), "device_count() < N with N > 1"),
    (re.compile(r"""torch\.device\(\s*["']cuda["']\s*,\s*[1-9]"""), 'torch.device("cuda", N) with N > 0'),
    (re.compile(r"""["']cuda:[1-9]\d*["']"""), "hard-coded 'cuda:N' with N > 0"),
    (re.compile(r"\bgpu_index\s*=\s*[1-9]"), "gpu_index = N with N > 0"),
)


@pytest.mark.parametrize("filename", AFFECTED_FILES)
def test_no_fixed_device_index_gate(filename):
    path = TESTS_DIR / filename
    assert path.is_file(), f"{filename} is missing"

    hits = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        for pattern, label in FORBIDDEN_PATTERNS:
            if pattern.search(line):
                hits.append(f"{filename}:{lineno}: {label}: {line.strip()}")

    assert not hits, (
        "Fixed device-index gates make these tests skip on ordinary hosts; "
        "select the device relative to what the host exposes (cuda:0 / "
        "torch.cuda.current_device()) and gate only on the required count:\n"
        + "\n".join(hits)
    )
