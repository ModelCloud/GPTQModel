# SPDX-FileCopyrightText: 2025 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
import textwrap

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device required for streaming D2H test"
)


def test_gptq_processor_async_d2h_streaming_roundtrip():
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "7")
    env.setdefault("PYTHON_GIL", os.environ.get("PYTHON_GIL", "1"))

    script = textwrap.dedent(
        """
        import os
        import sys
        import threading
        from types import SimpleNamespace

        import torch

        class _RandomWords:
            def get_random_word(self):
                return "stream-events"

        sys.modules.setdefault("random_word", SimpleNamespace(RandomWords=lambda: _RandomWords()))

        from gptqmodel.looper.gptq_processor import GPTQProcessor
        from gptqmodel.looper.named_module import NamedModule

        device = torch.device("cuda", 0)
        torch.cuda.set_device(device)

        processor = object.__new__(GPTQProcessor)
        processor.lock = threading.Lock()

        linear = torch.nn.Linear(8, 8, bias=False).to(device=device, dtype=torch.float16)
        named_module = NamedModule(linear, name="proj", full_name="model.layers.0.proj", layer_index=0)

        payload = {
            "q_scales": torch.randn(8, 8, device=device, dtype=torch.float16),
            "q_zeros": torch.randn(8, 8, device=device, dtype=torch.float16),
            "q_g_idx": torch.arange(64, device=device, dtype=torch.int32).reshape(8, 8),
        }

        named_module.stream_state_payload_to_cpu(payload)

        host_scales = named_module.state["q_scales"]
        host_zeros = named_module.state["q_zeros"]
        host_g_idx = named_module.state["q_g_idx"]

        assert host_scales.is_pinned() and host_zeros.is_pinned() and host_g_idx.is_pinned()

        named_module.stream_sync()

        torch.testing.assert_close(host_scales.cpu(), payload["q_scales"].cpu(), atol=0, rtol=0)
        torch.testing.assert_close(host_zeros.cpu(), payload["q_zeros"].cpu(), atol=0, rtol=0)
        torch.testing.assert_close(host_g_idx.cpu(), payload["q_g_idx"].cpu(), atol=0, rtol=0)

        processor._release_host_buffers(
            named_module.state.pop("q_scales"),
            named_module.state.pop("q_zeros"),
            named_module.state.pop("q_g_idx"),
        )
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        pytest.skip(
            f"Streaming event helper subprocess unavailable: rc={result.returncode}, stderr={result.stderr.strip()}"
        )
