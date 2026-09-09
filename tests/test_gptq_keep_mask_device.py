from types import SimpleNamespace

import pytest
import torch

from gptqmodel.looper.gptq_processor import GPTQProcessor
from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ


@pytest.mark.parametrize(
    "mask_device,input_device,output_device",
    [
        ("cpu", "cpu", "cpu"),
        pytest.param(
            "cuda:0",
            "cuda:0",
            "cuda:0",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is required"
            ),
        ),
        pytest.param(
            "cuda:0",
            "cuda:1",
            "cuda:1",
            marks=pytest.mark.skipif(
                torch.cuda.device_count() < 2, reason="Two CUDA devices are required"
            ),
        ),
        pytest.param(
            "cuda:0",
            "cuda:0",
            "cuda:1",
            marks=pytest.mark.skipif(
                torch.cuda.device_count() < 2, reason="Two CUDA devices are required"
            ),
        ),
    ],
)
def test_keep_mask_preserves_samples_across_devices(
    mask_device: str,
    input_device: str,
    output_device: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = torch.nn.Linear(2, 2, bias=False, device=input_device)
    task = GPTQ(layer, qcfg=QuantizeConfig(bits=4, group_size=2))
    processor = object.__new__(GPTQProcessor)
    processor.tasks = {"linear": task}
    processor._batch_tls = SimpleNamespace(index=7)
    processor._mask_tls = SimpleNamespace(
        value=torch.tensor(
            [[True, False, True], [False, True, False], [False, False, False]],
            device=mask_device,
        )
    )
    inputs = torch.arange(18, device=input_device, dtype=torch.float32).reshape(3, 3, 2)
    outputs = layer(inputs).to(output_device)
    captured = []
    add_batch = task.add_batch

    def capture(
        inp: torch.Tensor, out: torch.Tensor, batch_index: int | None = None
    ) -> None:
        captured.append((inp, out, batch_index))
        add_batch(inp, out, batch_index=batch_index)

    monkeypatch.setattr(task, "add_batch", capture)
    processor.pre_process_fwd_hook("linear")(layer, (inputs,), outputs)

    assert len(captured) == 2
    for (inp, out, batch_index), (row, positions) in zip(
        captured, [(0, [0, 2]), (1, [1])]
    ):
        torch.testing.assert_close(inp, inputs[row : row + 1, positions, :])
        torch.testing.assert_close(out, outputs[row : row + 1, positions, :])
        assert batch_index == 7
        assert inp.is_contiguous() and out.is_contiguous()
    kept = torch.stack([inputs[0, 0], inputs[0, 2], inputs[1, 1]])
    torch.testing.assert_close(task.finalize_hessian(), 2.0 * kept.T @ kept / 3)
    assert task.nsamples == 3
    assert task.fwd_counter == 2
