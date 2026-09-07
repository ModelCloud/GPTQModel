"""Exact native HIP window decode checks; no Triton imports or execution."""

import ctypes
import os
import unittest

import torch


@unittest.skipUnless(
    os.environ.get("QVQ_GFX950_DECODE_LIBRARY"), "supply experimental HIP library"
)
class NativeDecodeTest(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("QVQ_GFX950_ROCBLAS_LIBRARY"),
        "supply experimental rocBLAS library",
    )
    def test_rocblas_fp32_output_and_graph(self):
        lib = ctypes.CDLL(os.environ["QVQ_GFX950_ROCBLAS_LIBRARY"])
        lib.qvq_gfx950_rocblas_prepare.argtypes = (
            [ctypes.c_int] * 3
            + [ctypes.c_void_p] * 2
            + [ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)]
        )
        lib.qvq_gfx950_rocblas_execute.argtypes = [ctypes.c_void_p] * 5
        lib.qvq_gfx950_rocblas_destroy.argtypes = [ctypes.c_void_p]

        class Config(ctypes.Structure):
            _fields_ = [
                (name, ctypes.c_int32)
                for name in (
                    "struct_size",
                    "version",
                    "m",
                    "k",
                    "n",
                    "e",
                    "solution_index",
                    "reserved",
                )
            ]

        lib.qvq_gfx950_rocblas_solutions.argtypes = [ctypes.c_void_p] * 4 + [
            ctypes.POINTER(ctypes.c_int32)
        ] * 2
        lib.qvq_gfx950_rocblas_get_config.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(Config),
        ]
        lib.qvq_gfx950_rocblas_prepare_config.argtypes = (
            [ctypes.POINTER(Config)]
            + [ctypes.c_void_p] * 5
            + [ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p)]
        )
        gen = torch.Generator(device="cuda").manual_seed(951)
        for m, k, n in ((1, 256, 256), (8, 5120, 1024), (128, 5120, 1024)):
            x = (
                torch.randn((m, k), generator=gen, device="cuda", dtype=torch.float16)
                * 0.1
            )
            w = (
                torch.randn((n, k), generator=gen, device="cuda", dtype=torch.float16)
                * 0.1
            )
            y = torch.empty((m, n), device="cuda", dtype=torch.float32)
            workspace = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
            plan = ctypes.c_void_p()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            graph = None
            with torch.cuda.stream(stream):
                self.assertEqual(
                    lib.qvq_gfx950_rocblas_prepare(
                        m,
                        k,
                        n,
                        stream.cuda_stream,
                        workspace.data_ptr(),
                        workspace.numel(),
                        ctypes.byref(plan),
                    ),
                    0,
                )
                try:
                    count = ctypes.c_int32()
                    self.assertEqual(
                        lib.qvq_gfx950_rocblas_solutions(
                            plan,
                            x.data_ptr(),
                            w.data_ptr(),
                            y.data_ptr(),
                            None,
                            ctypes.byref(count),
                        ),
                        0,
                    )
                    self.assertGreater(count.value, 0)
                    solutions = (ctypes.c_int32 * count.value)()
                    self.assertEqual(
                        lib.qvq_gfx950_rocblas_solutions(
                            plan,
                            x.data_ptr(),
                            w.data_ptr(),
                            y.data_ptr(),
                            solutions,
                            ctypes.byref(count),
                        ),
                        0,
                    )
                    self.assertTrue(all(solution >= 0 for solution in solutions))
                    config = Config(
                        ctypes.sizeof(Config), 1, m, k, n, 1, solutions[0], 0
                    )
                    selected = ctypes.c_void_p()
                    self.assertEqual(
                        lib.qvq_gfx950_rocblas_prepare_config(
                            ctypes.byref(config),
                            x.data_ptr(),
                            w.data_ptr(),
                            y.data_ptr(),
                            stream.cuda_stream,
                            workspace.data_ptr(),
                            workspace.numel(),
                            ctypes.byref(selected),
                        ),
                        0,
                    )
                    self.assertEqual(lib.qvq_gfx950_rocblas_destroy(plan), 0)
                    plan = selected
                    resolved = Config(ctypes.sizeof(Config), 1)
                    self.assertEqual(
                        lib.qvq_gfx950_rocblas_get_config(plan, ctypes.byref(resolved)),
                        0,
                    )
                    self.assertEqual(bytes(resolved), bytes(config))
                    print(
                        f"MKNE={m},{k},{n},1: {count.value} solutions; explicit solution {config.solution_index}"
                    )
                    args = (
                        plan,
                        x.data_ptr(),
                        w.data_ptr(),
                        y.data_ptr(),
                        stream.cuda_stream,
                    )
                    self.assertEqual(lib.qvq_gfx950_rocblas_execute(*args), 0)
                    stream.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        self.assertEqual(lib.qvq_gfx950_rocblas_execute(*args), 0)
                    for _ in range(3):
                        x.mul_(0.5)
                        graph.replay()
                        expected = x.double() @ w.double().T
                        error = (y.double() - expected).abs()
                        self.assertTrue(torch.isfinite(error).all())
                        self.assertLessEqual(error.mean().item(), 0.003)
                        self.assertLessEqual(error.max().item(), 0.006)
                    stream.synchronize()
                finally:
                    stream.synchronize()
                    del graph
                    self.assertEqual(lib.qvq_gfx950_rocblas_destroy(plan), 0)

    def test_all_rates_banks_and_replay(self):
        lib = ctypes.CDLL(os.environ["QVQ_GFX950_DECODE_LIBRARY"])
        op = lib.qvq_gfx950_decode_window
        op.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_uint] * 4 + [ctypes.c_void_p]
        op.restype = ctypes.c_int
        self.assertEqual(op(None, None, None, None, 16, 16, 6, 1, None), 1)
        self.assertTrue(torch.version.hip)
        self.assertEqual(
            torch.cuda.get_device_properties(0).gcnArchName.split(":")[0], "gfx950"
        )
        masks = (
            (0, 0x5A5A, 0x3C3C, 0xC3C3),
            (0, 0x9696, 0x3C3C, 0xC3C3),
            (0, 0x6969, 0x5A5A, 0x3C3C),
            (0, 0xC3C3, 0x9696, 0x5A5A),
        )
        gen = torch.Generator().manual_seed(950)
        # Unique exactly representable LUT entries expose swapped byte indices.
        levels_cpu = torch.arange(256, dtype=torch.float16) / 256
        levels = levels_cpu.cuda()
        cases = 0
        for k, n in ((16, 16), (32, 48), (256, 256)):
            tiles = k * n // 256
            for bits in range(4, 8):
                words = torch.randint(
                    0, 2**32, (tiles, 4 * bits), generator=gen, dtype=torch.int64
                )
                banks_cpu = torch.randint(
                    0, 256, (tiles,), generator=gen, dtype=torch.uint8
                )
                window = words.to(torch.int32).cuda()
                banks = banks_cpu.cuda()
                output = torch.empty((n, k), dtype=torch.float16, device="cuda")
                for bank in range(4):
                    # Independent arbitrary-width circular bitstream oracle.
                    expected = torch.empty((n, k), dtype=torch.float16)
                    for tile, row_words in enumerate(words.tolist()):
                        stream_bits = sum(
                            word << (32 * index) for index, word in enumerate(row_words)
                        )
                        circular = stream_bits | (stream_bits << (128 * bits))
                        for pair in range(128):
                            state = (circular >> ((127 - pair) * bits)) & 65535
                            state ^= (
                                (int(banks_cpu[tile]) >> (pair // 16)) & 1
                            ) * masks[bits - 4][bank]
                            mixed = ((state ^ (state >> 8)) * 40503 + 17011) & 65535
                            mixed ^= mixed >> 7
                            col = (tile % (n // 16)) * 16 + (pair % 8) * 2
                            row = (tile // (n // 16)) * 16 + pair // 8
                            expected[col, row] = levels_cpu[mixed >> 8]
                            expected[col + 1, row] = levels_cpu[mixed & 255]

                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        args = (
                            window.data_ptr(),
                            levels.data_ptr(),
                            banks.data_ptr(),
                            output.data_ptr(),
                            k,
                            n,
                            bits,
                            bank,
                            stream.cuda_stream,
                        )

                        def run(args=args):
                            self.assertEqual(op(*args), 0)

                        run()
                        stream.synchronize()
                        self.assertTrue(torch.equal(output.cpu(), expected))
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, stream=stream):
                            run()
                        for scale in (0.5, 2.0, 1.0):
                            levels.copy_(levels_cpu.cuda() * scale)
                            graph.replay()
                            stream.synchronize()
                            self.assertTrue(torch.equal(output.cpu(), expected * scale))
                        del graph
                    cases += 1
        print(
            f"Exact decode: {cases} rate/bank/shape cases; three changed-LUT replays each"
        )


if __name__ == "__main__":
    unittest.main()
