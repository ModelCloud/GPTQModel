# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

import builtins
import sys

import pytest

from gptqmodel.models.definitions import base_qwen3_vl


def test_qwen3_vl_video_missing_dependency_has_install_hint(monkeypatch):
    monkeypatch.delitem(sys.modules, "qwen_vl_utils", raising=False)
    monkeypatch.delitem(sys.modules, "qwen_vl_utils.vision_process", raising=False)

    real_import = builtins.__import__

    def fail_qwen_vl_import(name, *args, **kwargs):
        if name.startswith("qwen_vl_utils"):
            raise ImportError("No module named 'qwen_vl_utils'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_qwen_vl_import)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": "dummy.mp4"},
                {"type": "text", "text": "Describe this video."},
            ],
        }
    ]

    with pytest.raises(ImportError, match="pip install qwen-vl-utils"):
        base_qwen3_vl.BaseQwen3VLGPTQ.process_vision_info(messages)
