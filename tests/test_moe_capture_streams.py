# SPDX-License-Identifier: Apache-2.0

import threading

import pytest
import torch
import torch.nn.functional as F

from gptqmodel.models.moe_capture_streams import RoutedMoECaptureStreamAttachment


def test_capture_stream_attachment_serial_fallback_preserves_order():
    attachment = RoutedMoECaptureStreamAttachment()
    observed = []
    results = attachment.run(
        device=torch.device("cpu"),
        entries=[3, 1, 2],
        stream_count=2,
        launch=lambda value: observed.append(value) or value * 2,
    )
    assert observed == [3, 1, 2]
    assert results == [6, 2, 4]


@pytest.mark.parametrize("seed", [20260810, 20260811, 20260812])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_capture_stream_attachment_matches_serial_expert_hessian_bitwise(seed):
    torch.manual_seed(seed)
    device = torch.device("cuda:0")
    expert_count = 4
    tokens, hidden_size, intermediate_size = 64, 128, 64
    hidden = torch.randn(tokens, hidden_size, device=device, dtype=torch.bfloat16)
    gate_weights = [
        torch.randn(intermediate_size, hidden_size, device=device, dtype=torch.bfloat16)
        for _ in range(expert_count)
    ]
    up_weights = [
        torch.randn(intermediate_size, hidden_size, device=device, dtype=torch.bfloat16)
        for _ in range(expert_count)
    ]

    attachment = RoutedMoECaptureStreamAttachment()

    def collect(stream_count):
        hessians = [
            torch.zeros(intermediate_size, intermediate_size, device=device)
            for _ in range(expert_count)
        ]

        def launch(index):
            gate = F.linear(hidden, gate_weights[index])
            up = F.linear(hidden, up_weights[index])
            intermediate = F.silu(gate) * up
            materialized = intermediate.float()
            hessians[index].addmm_(materialized.T, materialized)

        attachment.run(
            device=device,
            entries=list(range(expert_count)),
            stream_count=stream_count,
            launch=launch,
        )
        torch.cuda.synchronize(device)
        return hessians

    serial = collect(1)
    for _ in range(10):
        parallel = collect(2)
        for expected, actual in zip(serial, parallel):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_capture_stream_attachment_is_bitwise_exact_with_one_free_threaded_owner_per_device():
    attachment = RoutedMoECaptureStreamAttachment()
    barrier = threading.Barrier(2)
    failures = []

    def run_device(device_index):
        try:
            device = torch.device(f"cuda:{device_index}")
            torch.cuda.set_device(device)
            generator = torch.Generator(device=device).manual_seed(20260820 + device_index)
            hidden = torch.randn((48, 96), generator=generator, device=device, dtype=torch.bfloat16)
            gate_weights = [
                torch.randn((64, 96), generator=generator, device=device, dtype=torch.bfloat16)
                for _ in range(4)
            ]
            up_weights = [
                torch.randn((64, 96), generator=generator, device=device, dtype=torch.bfloat16)
                for _ in range(4)
            ]

            def collect(stream_count):
                hessians = [torch.zeros((64, 64), device=device) for _ in gate_weights]

                def launch(index):
                    intermediate = F.silu(F.linear(hidden, gate_weights[index])) * F.linear(
                        hidden,
                        up_weights[index],
                    )
                    values = intermediate.float()
                    hessians[index].addmm_(values.T, values)

                attachment.run(
                    device=device,
                    entries=range(len(gate_weights)),
                    stream_count=stream_count,
                    launch=launch,
                )
                torch.cuda.synchronize(device)
                return hessians

            serial = collect(1)
            barrier.wait(timeout=10)
            for _ in range(5):
                parallel = collect(2)
                for expected, actual in zip(serial, parallel):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        except BaseException as exc:
            failures.append(exc)

    threads = [threading.Thread(target=run_device, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    assert all(not thread.is_alive() for thread in threads)
    assert not failures
