# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Raw Metal kernels lowered into the TT-XLA PJRT executable."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache, wraps
from pathlib import Path

import torch

from .nf12_tenstorrent import (
    LINEAR_PACKED_BUFFER_DEPTH,
    LINEAR_PACKED_SCRATCH_BYTES,
    kernel_source,
    normalize_ids,
    round_up,
    verify_nf12,
)


_PREFIX = "gptqmodel.nf12.v1."
_ARTIFACT_VERSION = 2
_GRID_X, _GRID_Y = 8, 8


def _tiled_words(words):
    """Expose the original stream through PJRT's face-ordered INT32 tiles."""
    flat = words.reshape(-1).view(torch.int32)
    size = max(1024, round_up(flat.numel(), 1024))
    flat = torch.nn.functional.pad(flat, (0, size - flat.numel()))
    return flat.reshape(-1, 2, 2, 16, 16).transpose(2, 3).reshape(-1, 32)


def _cb(index, data_format, page_size, count=1):
    return {"buffer_index": index, "data_format": data_format, "page_size": page_size, "total_size": count * page_size}


def _artifact(operation_id, version_tag, shapes, dtypes, layouts, mesh_shape, arg_roles, **_unused):
    fields = operation_id.removeprefix(_PREFIX).split(".")
    if len(fields) != 7:
        raise ValueError(f"Invalid TT-XLA NF12 operation ID {operation_id!r}: expected seven fields")
    op = fields[0]
    n, k, m, base, m_block, flag = (
        int(fields[1]), int(fields[2]), int(fields[3]), int(fields[4]), int(fields[5]), int(fields[6]),
    )
    if tuple(mesh_shape) != (1, 1) or any(
        "<interleaved>" not in layout or "<dram>" not in layout or "!ttcore.tile<32x32" not in layout
        for layout in layouts
    ):
        raise ValueError("NF12 TT-XLA requires one chip and interleaved 32x32 DRAM tiles")
    if op == "gather":
        expected = [(max(1024, round_up(flag * k * 3 // 8, 1024)) // 32, 32),
                    tuple(shapes[1]), (1, round_up(m, 32)), (round_up(m, 32), k)]
        expected_dtypes = ("si32", "si32", "si32", "bf16")
        tasks = round_up(m, 32) // 32 * (k // 32)
        cbs = [_cb(24, "UInt32", 1536 + 128 + 2048 + 8 * 64)]
        kernels = [("gather_xla", "noc", [0, 1, 2, 3], [k, tasks])]
    elif op == "linear":
        nt, kt, mt = round_up(n, 32) // 32, k // 32, round_up(m, 32) // 32
        dense, wide = bool(flag & 1), bool(flag & 2)
        w_shape = (nt * kt * 32, 32) if dense else (round_up(n * k * 3 // 8, 1024) // 32, 32)
        expected = [(mt * 32, k), w_shape, tuple(shapes[2]), (mt * 32, nt * 32)]
        expected_dtypes = ("bf16", "bf16" if dense else "si32", "si32", "f32" if wide else "bf16")
        tasks = nt
        out_page = 4096 if wide else 2048
        depth = 2 if dense else LINEAR_PACKED_BUFFER_DEPTH
        cbs = [
            _cb(0, "BFloat16", 2048, depth),
            _cb(1, "BFloat16", 2048, depth * m_block),
            _cb(16, "Float32" if wide else "BFloat16", out_page, 2),
        ]
        if not dense:
            cbs.append(_cb(24, "UInt32", LINEAR_PACKED_SCRATCH_BYTES))
        reader_name = "linear_reader" if dense else "linear_pair_reader"
        kernels = [
            (reader_name, "noc", [0, 1, 2], [n // 16, kt, mt, m_block, base, int(dense), nt]),
            ("linear_writer", "noc", [3], [nt, mt, out_page]),
            ("linear_compute", "compute", [], [kt, mt, m_block, nt]),
        ]
    elif op == "sfpu_linear":
        nt, kt = n // 32, k // 32
        mt = round_up(m, 32) // 32
        dense = bool(flag)
        metadata = tuple(shapes[2])
        if (m <= 0 or n <= 0 or n % 32 or k <= 0 or k % 128 or m_block < 1 or kt // 4 % m_block
                or not 0 <= base <= 120 or flag not in (0, 1) or len(metadata) != 2
                or metadata[1] != 32 or metadata[0] % 128 != base
                or metadata[0] * 32 < round_up(nt * kt // 2, 16)):
            raise ValueError("Unexpected TT-XLA SFPU LM-head geometry or exponent base")
        w_shape = (nt * kt * 32, 32) if dense else (round_up(nt * (kt // 4) * 1536, 1024) // 32, 32)
        expected = [(mt * 32, k), w_shape, metadata, (mt * 32, n)]
        expected_dtypes = ("bf16", "bf16" if dense else "si32", "si32", "bf16")
        tasks = nt
        cbs = [
            _cb(1, "BFloat16", 2048, kt),
            _cb(2, "BFloat16", 2048, kt if dense else 1),
            _cb(17, "BFloat16", 2048, 2),
        ]
        if not dense:
            cbs += [
                _cb(0, "UInt16", 2048, min(m_block * 6, kt // 4 * 3)),
                _cb(16, "UInt16", 2048, kt),
                _cb(24, "UInt32", 128),
            ]
        kernels = [
            ("sfpu_linear_reader", "noc", [1, 0], [kt, int(dense), m_block, nt, mt]),
            ("sfpu_linear_writer", "noc", [3] if dense else [3, 2], [kt, int(dense), int(not dense), nt, mt]),
            ("sfpu_linear_compute", "compute", [], [kt, int(dense), base, nt, mt]),
        ]
    else:
        raise ValueError(f"Unknown TT-XLA NF12 operation {op}")
    if (list(map(tuple, shapes)) != expected or tuple(dtypes) != expected_dtypes
            or arg_roles != "in,in,in,out" or version_tag != _version_tag()):
        raise ValueError(f"Unexpected TT-XLA NF12 operand or kernel version for {operation_id}")

    grid_x = min(tasks, 11 if op == "sfpu_linear" else _GRID_X)
    grid_y = min(max(1, tasks // grid_x), 10 if op == "sfpu_linear" else _GRID_Y)
    sources = []
    for name, thread, indices, config in kernels:
        if thread == "compute":
            kernel_config = {"type": "ComputeKernelConfig", "math_fidelity": "HiFi4", "fp32_dest_acc_en": True,
                             "dst_full_sync_en": False, "bfp8_pack_precise": False, "math_approx_mode": False}
        else:
            kernel_config = {"type": "WriterKernelConfig" if name.endswith("_writer") else "ReaderKernelConfig"}
        prelude = (
            "#define NF12_PJRT 1\n"
            f"constexpr unsigned int nf12_cb_count = {len(cbs)};\n"
            f"constexpr unsigned int nf12_grid_x = {grid_x};\n"
            f"constexpr unsigned int nf12_core_count = {grid_x * grid_y};\n"
            f"constexpr unsigned int nf12_config[] = {{{', '.join(map(str, config))}}};\n"
        )
        sources.append({"thread_type": thread, "cpp_source": prelude + kernel_source(name),
                            "tensor_indices": indices, "kernel_config": kernel_config})
    return json.dumps({
        "format_version": _ARTIFACT_VERSION, "kernels": sources, "cb_configs": cbs, "num_tensors": 4,
        "core_range": {"start": [0, 0], "end": [grid_x - 1, grid_y - 1]},
        "num_pipe_sync_semaphores": 0, "num_pipe_global_semaphores": 0, "pipe_sram_scratch_bytes": 0,
    }).encode()


@lru_cache(maxsize=1)
def _version_tag():
    digest = hashlib.sha256()
    digest.update(Path(__file__).read_bytes())
    for name in ("gather_xla", "linear_reader", "linear_pair_reader", "linear_compute", "linear_writer",
                 "sfpu_linear_reader", "sfpu_linear_writer", "sfpu_linear_compute"):
        digest.update(kernel_source(name).encode())
    return digest.hexdigest()[:24]


def _install_resolver():
    # TT-XLA's tt-lang bridge owns the compiler hook. Delegate all operations
    # outside GPT-QModel's NF12 namespace to its original resolver.
    from tt_torch import tt_lang

    old = tt_lang.resolve_operation
    if getattr(old, "_gptqmodel_nf12", False):
        return
    if getattr(tt_lang, "_ARTIFACT_FORMAT_VERSION", None) != _ARTIFACT_VERSION:
        raise RuntimeError("This TT-XLA version does not expose NF12's kernel artifact ABI")

    @wraps(old)
    def resolve(**kwargs):
        if kwargs["operation_id"].startswith(_PREFIX):
            return _artifact(**kwargs)
        return old(**kwargs)

    resolve._gptqmodel_nf12 = True
    tt_lang.resolve_operation = resolve


def _call(op, config, inputs, out):
    from tt_torch.custom_ops import tt_lang_op_dispatch

    op_id = _PREFIX + op + "." + ".".join(map(str, config))
    return tt_lang_op_dispatch(
        [*inputs, out], kernel_id=op_id, arg_roles="in,in,in,out", version_tag=_version_tag(),
        shard_spec="", out_indices=[3],
    )[0]


def _blackhole_device(device):
    import torch_xla.runtime as xr

    device = torch.device(device)
    if device.type != "xla" or xr.device_type() != "TT":
        raise ValueError("TT-XLA NF12 requires the Tenstorrent PJRT device")
    arch = xr.runtime_device_attributes(str(device)).get("device_arch", "")
    if str(arch).casefold() != "blackhole":
        raise ValueError(f"TT-XLA NF12 requires a Blackhole chip, received {arch!r}")
    return device


class TTXLANF12:
    """Single-chip packed weights for the Tenstorrent PJRT backend."""

    def __init__(self, device, descriptor, weight):
        device = _blackhole_device(device)
        verify_nf12(descriptor, weight)
        _install_resolver()
        self.device = device
        self.descriptor = descriptor
        self.packed = _tiled_words(weight.packed.view(torch.int32)).to(device)
        # The Torch/XLA lazy cache can reuse an executable for equal operand
        # shapes even when frontend attributes differ. Read per-weight values
        # from memory so two matrices can safely use the same executable.
        header = torch.zeros(16, dtype=torch.int32)
        header[:3] = torch.tensor([weight.exp_base, descriptor.logical_shape[0], weight.shape[0] // 16])
        escape_stream = torch.cat((header, weight.escapes.view(torch.int32).flatten()))
        self.escapes = _tiled_words(escape_stream).to(device)

    @property
    def resident_bytes(self):
        return (self.packed.numel() + self.escapes.numel()) * 4

    def upload_ids(self, ids, padding_idx=None):
        if ids.device.type != "cpu" and ids.device != self.device:
            raise ValueError("TT-XLA NF12 IDs must be CPU tensors or on the packed table's device")
        if ids.dtype == torch.int64 and ids.device.type == "xla":
            raise ValueError("TT-XLA NF12 expects int32 device IDs or int64 CPU IDs")
        flat = normalize_ids(ids, self.descriptor.logical_shape[0], padding_idx).reshape(-1)
        if not flat.numel():
            return flat[None, :]
        return torch.nn.functional.pad(flat, (0, round_up(flat.numel(), 32) - flat.numel()), value=-1)[None, :].to(self.device)

    def gather_device(self, ids, count):
        """Gather from a padded int32 ID tensor already resident on the device."""
        if count <= 0:
            raise ValueError("NF12 expects at least one ID")
        n, k = self.descriptor.logical_shape
        padded_n, padded_k = self.descriptor.padded_shape
        rows = round_up(count, 32)
        if ids.device != self.device or ids.dtype != torch.int32 or tuple(ids.shape) != (1, rows):
            raise ValueError("TT-XLA NF12 expects padded int32 IDs on the packed table's device")
        out = torch.empty((rows, padded_k), dtype=torch.bfloat16, device=self.device)
        result = _call("gather", (n, padded_k, count, 0, 0, padded_n),
                       (self.packed, self.escapes, ids), out)
        return result[:count, :k]

    def embedding(self, ids, padding_idx=None):
        k = self.descriptor.logical_shape[1]
        resident = self.upload_ids(ids, padding_idx)
        if not ids.numel():
            return torch.empty((*ids.shape, k), dtype=torch.bfloat16, device=self.device)
        return self.gather_device(resident, ids.numel()).reshape(*ids.shape, k)

    def upload_baseline(self, w):
        n, k = self.descriptor.logical_shape
        padded_n, padded_k = self.descriptor.padded_shape
        if w.shape != (n, k) or w.dtype != torch.bfloat16:
            raise ValueError("Matched baseline expects the original BF16 weight")
        padded = torch.nn.functional.pad(w, (0, padded_k - k, 0, round_up(padded_n, 32) - n))
        tiles = padded.reshape(-1, 32, padded_k // 32, 32).permute(0, 2, 3, 1).contiguous()
        return tiles.reshape(-1, 32).to(self.device)

    def linear(self, x, bias=None, *, baseline=None, m_block=4):
        n, k = self.descriptor.logical_shape
        padded_n, padded_k = self.descriptor.padded_shape
        if x.device != self.device or x.dtype != torch.bfloat16 or not x.ndim or x.shape[-1] != k:
            raise ValueError("TT-XLA NF12 expects BF16 activations on the packed weight's device")
        if bias is not None and (bias.shape != (n,) or bias.device != self.device
                                 or bias.dtype not in (torch.bfloat16, torch.float32)):
            raise ValueError("TT-XLA NF12 bias must be BF16 or FP32 [N] on the packed weight's device")
        if not 1 <= m_block <= 4:
            raise ValueError("NF12 M block must be in [1, 4]")
        if baseline is not None and (baseline.shape != (round_up(padded_n, 32) * padded_k // 32, 32)
                                     or baseline.dtype != torch.bfloat16 or baseline.device != self.device):
            raise ValueError("Unexpected TT-XLA NF12 baseline shape, dtype, or device")
        if not x.numel():
            return x.new_empty((*x.shape[:-1], n))
        rows = x.reshape(-1, k)
        m = rows.shape[0]
        mt, nt = round_up(m, 32), round_up(padded_n, 32)
        activation = torch.nn.functional.pad(rows, (0, padded_k - k, 0, mt - m))
        dtype = torch.bfloat16 if bias is None else torch.float32
        out = torch.empty((mt, nt), dtype=dtype, device=self.device)
        flags = int(baseline is not None) | (2 * int(bias is not None))
        weight = self.packed if baseline is None else baseline
        block = min(m_block, mt // 32)
        # A leading size-one dimension distinguishes M-reuse schedules in
        # Torch/XLA's cache without adding a resident operand or changing pages.
        escapes = self.escapes.reshape((1,) * (block - 1) + tuple(self.escapes.shape))
        y = _call("linear", (padded_n, padded_k, m, 0, block, flags),
                  (activation, weight, escapes), out)[:m, :n]
        if bias is not None:
            y = (y + bias.float()).to(torch.bfloat16)
        return y.reshape(*x.shape[:-1], n)
