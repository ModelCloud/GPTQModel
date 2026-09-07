"""Public runtime parity; requires an explicitly supplied QVQ AOT artifact."""

import ctypes as c
import hashlib
import json
import os
from pathlib import Path

import pytest
import torch


@pytest.mark.cuda
def test_public_gfx950_abi_eager_and_graph():
    artifact = os.environ.get("QVQ_GFX950_TEST_ARTIFACT")
    library = os.environ.get("QVQ_GFX950_TEST_LIBRARY")
    if not artifact or not library:
        pytest.skip("set QVQ_GFX950_TEST_ARTIFACT and QVQ_GFX950_TEST_LIBRARY")
    if not torch.cuda.is_available() or not torch.version.hip:
        pytest.skip("requires ROCm")
    from gptqmodel.quantization.qvq import (
        reconstruct_qvq_inner_weight,
        repack_p32_planar_to_window,
    )
    from gptqmodel.quantization.qvq_codecs import (
        PGC16_CODEBOOK_VERSION,
        pgc16_levels_for_version,
    )
    from gptqmodel.utils.qvq_amd import qvq_p32_amd

    path = Path(artifact)
    record = json.loads((path / "manifest.json").read_text())
    payload = (path / "kernel.hsaco").read_bytes()
    assert hashlib.sha256(payload).hexdigest() == record["sha256"]
    assert record["arch"] == "gfx950"
    fields = (
        "abi_version",
        "operation_version",
        "m",
        "k",
        "n",
        "transition_bits",
        "bank_alt_id",
        "grid_x",
        "threads",
        "shared_bytes",
    )

    class Spec(c.Structure):
        _fields_ = [(name, c.c_uint32) for name in fields]

    lib = c.CDLL(library)
    lib.qvq_gfx950_last_error.restype = c.c_char_p
    lib.qvq_gfx950_prepare.argtypes = [
        c.POINTER(Spec),
        c.c_void_p,
        c.c_size_t,
        c.c_char_p,
        c.c_int,
        c.c_void_p,
        c.POINTER(c.c_void_p),
    ]
    lib.qvq_gfx950_execute.argtypes = [c.c_void_p] * 7
    lib.qvq_gfx950_destroy.argtypes = [c.c_void_p]

    def check(result):
        assert result == 0, lib.qvq_gfx950_last_error().decode()

    m, k, n = (record[key] for key in ("m", "k", "n"))
    bits = record["transition_bits"] / 2
    generator = torch.Generator(device="cuda").manual_seed(950)
    x = (
        torch.randn((m, k), generator=generator, device="cuda", dtype=torch.float16)
        * 0.1
    )
    planar = torch.randint(
        -(1 << 31),
        1 << 31,
        (k * n // 256, record["transition_bits"] * 4),
        generator=generator,
        device="cuda",
        dtype=torch.int32,
    )
    if bits == 4:
        # W4 has one 8-bit plane: reverse chronological bytes into windows.
        window = planar.view(torch.uint8).flip(-1).contiguous().view(torch.int32)
    else:
        window = repack_p32_planar_to_window(planar, bits=bits)
    banks = torch.randint(
        0, 256, (k * n // 256,), generator=generator, device="cuda", dtype=torch.uint8
    )
    levels = pgc16_levels_for_version(PGC16_CODEBOOK_VERSION).to("cuda")
    if bits == 4:
        banks.zero_()
    dense = reconstruct_qvq_inner_weight(
        planar,
        bits=bits,
        in_features=k,
        out_features=n,
        bank_ids=None if bits == 4 else banks,
        v2b2_p32=bits != 4,
        bank_alt_id=None
        if bits == 4
        else torch.tensor([record["bank_alt_id"]], device="cuda", dtype=torch.uint8),
    ).float()
    output = torch.empty((m, n), device="cuda", dtype=torch.float32)
    spec = Spec(*(record[key] for key in fields))
    image = c.create_string_buffer(payload)
    plan = c.c_void_p()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = None
    with torch.cuda.stream(stream):
        check(
            lib.qvq_gfx950_prepare(
                c.byref(spec),
                image,
                len(payload),
                record["symbol"].encode(),
                torch.cuda.current_device(),
                stream.cuda_stream,
                c.byref(plan),
            )
        )
        try:

            def run():
                check(
                    lib.qvq_gfx950_execute(
                        plan,
                        x.data_ptr(),
                        window.data_ptr(),
                        levels.data_ptr(),
                        banks.data_ptr(),
                        output.data_ptr(),
                        stream.cuda_stream,
                    )
                )

            def reference():
                if bits == 4:
                    return x.float() @ dense
                return qvq_p32_amd(
                    x,
                    window,
                    levels,
                    banks,
                    bits,
                    out_features=n,
                    bank_alt_id=record["bank_alt_id"],
                    cache_weight=False,
                )

            def assert_accuracy(expected):
                if bits != 4:
                    assert torch.equal(output, expected)
                error = (output.double() - (x.float() @ dense).double()).abs()
                assert torch.isfinite(error).all()
                assert error.mean().item() <= 0.003
                assert error.max().item() <= 0.006

            expected = reference()
            run()
            stream.synchronize()
            assert_accuracy(expected)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                run()
            for _ in range(3):
                x.mul_(-0.5)
                expected = reference()
                graph.replay()
                stream.synchronize()
                assert_accuracy(expected)
        finally:
            stream.synchronize()
            if graph is not None:
                graph.reset()
            check(lib.qvq_gfx950_destroy(plan))
