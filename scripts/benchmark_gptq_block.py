"""Compare eager and native CUDA GPTQ block and full quantization timings.

Run with a CUDA compiler available to PyTorch, for example CUDA_HOME set to
the matching CUDA toolkit. Compilation happens before any measured trial.
"""

import argparse
import os
import statistics
import time

import torch

from gptqmodel.quantization.config import QuantizeConfig
from gptqmodel.quantization.gptq import GPTQ
from gptqmodel.quantization.gptq_cuda import block_update_available, gptq_block_update


def _time_cuda(fn):
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    return time.perf_counter() - started, result


def _block_inputs(rows, group_size, bits):
    torch.manual_seed(117)
    work = torch.randn(rows, 128, device="cuda", dtype=torch.float32)
    hinv = torch.triu(torch.randn(128, 128, device="cuda") * 0.001)
    hinv.diagonal().fill_(1.0)
    effective_group_size = 128 if group_size == -1 else group_size
    column_groups = [column // effective_group_size for column in range(128)]
    group_count = column_groups[-1] + 1
    scale = torch.rand(group_count, rows, device="cuda") * 0.1 + 0.01
    zero = torch.full_like(scale, 1 << (bits - 1))
    return work, hinv, scale, zero, column_groups


def _block_run(native, inputs, bits):
    source, hinv, scale, zero, column_groups = inputs
    work = source.clone()
    quantized = torch.empty_like(work)
    errors = torch.empty_like(work)
    losses = torch.empty_like(work)
    if native:
        column_group = torch.tensor(column_groups, device="cuda", dtype=torch.int32)
        gptq_block_update(
            work, hinv, scale, zero, column_group,
            quantized, errors, losses, (1 << bits) - 1,
        )
    else:
        for column in range(128):
            weight = work[:, column]
            diag = hinv[column, column]
            row_scale = scale[column_groups[column]]
            row_zero = zero[column_groups[column]]
            q = row_scale * (
                torch.clamp(torch.round(weight / row_scale) + row_zero, 0, (1 << bits) - 1)
                - row_zero
            )
            quantized[:, column] = q
            delta = errors[:, column]
            torch.sub(weight, q, out=delta)
            losses[:, column] = delta.square() / diag.square()
            delta.div_(diag)
            work[:, column:] -= delta.unsqueeze(1) * hinv[column, column:]
    return work, quantized, errors, losses


def _full_run(native, weight, inputs, group_size, bits):
    if native:
        os.environ.pop("GPTQMODEL_DISABLE_GPTQ_CUDA", None)
    else:
        os.environ["GPTQMODEL_DISABLE_GPTQ_CUDA"] = "1"
    rows, columns = weight.shape
    layer = torch.nn.Linear(columns, rows, bias=False, device="cuda")
    layer.weight.data.copy_(weight)
    cfg = QuantizeConfig(bits=bits, group_size=group_size, sym=True, act_group_aware=group_size > 0)
    task = GPTQ(layer, qcfg=cfg)
    task.quantizer.configure(perchannel=True)
    task.add_batch(inputs, None)
    return task.quantize(blocksize=128)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--columns", type=int, default=2048)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--bits", type=int, choices=range(2, 9), default=4)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required")
    if not block_update_available():
        raise RuntimeError("Native CUDA extension could not be loaded")

    block_inputs = _block_inputs(args.rows, args.group_size, args.bits)
    block_times = {}
    block_outputs = {}
    for label, native in (("eager", False), ("cuda", True)):
        _time_cuda(lambda: _block_run(native, block_inputs, args.bits))
        measurements = [_time_cuda(lambda: _block_run(native, block_inputs, args.bits)) for _ in range(args.trials)]
        block_times[label] = statistics.median(elapsed for elapsed, _ in measurements)
        block_outputs[label] = measurements[-1][1]
    for lhs, rhs in zip(block_outputs["cuda"], block_outputs["eager"]):
        torch.testing.assert_close(lhs, rhs, rtol=0, atol=0)

    torch.manual_seed(119)
    weight = torch.randn(args.rows, args.columns, device="cuda")
    inputs = torch.randn(1, 256, args.columns, device="cuda")
    full_times = {}
    full_outputs = {}
    for label, native in (("eager", False), ("cuda", True)):
        measurements = [
            _time_cuda(lambda: _full_run(native, weight, inputs, args.group_size, args.bits))
            for _ in range(args.trials)
        ]
        full_times[label] = statistics.median(elapsed for elapsed, _ in measurements)
        full_outputs[label] = measurements[-1][1]
    for index in (0, 1, 2, 3):
        torch.testing.assert_close(full_outputs["cuda"][index], full_outputs["eager"][index], rtol=0, atol=0)
    assert full_outputs["cuda"][5] == full_outputs["eager"][5]

    print(
        f"device={torch.cuda.get_device_name()} rows={args.rows} "
        f"columns={args.columns} group_size={args.group_size} bits={args.bits} trials={args.trials}"
    )
    for title, times in (("block_update", block_times), ("total_quantize", full_times)):
        print(f"{title}: eager={times['eager']:.6f}s cuda={times['cuda']:.6f}s speedup={times['eager'] / times['cuda']:.2f}x")
    print("parity=bitwise_equal")


if __name__ == "__main__":
    main()
