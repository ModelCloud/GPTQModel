"""Host descriptor checks for the experimental native operation ABI."""

import ctypes as c
import os
import unittest


class BlasConfig(c.Structure):
    _fields_ = [
        (name, c.c_int32)
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


class NativeConfig(c.Structure):
    _fields_ = [
        ("struct_size", c.c_uint32),
        ("version", c.c_uint32),
        ("gemm", BlasConfig),
    ] + [
        (name, c.c_uint32)
        for name in ("transition_bits", "bank_alt_id", "decode_threads", "cache_policy")
    ]


@unittest.skipUnless(
    os.environ.get("QVQ_GFX950_NATIVE_LIBRARY"), "supply native library"
)
class NativePlanHostTests(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("QVQ_GFX950_NATIVE_GPU_TEST"), "opt in to gfx950 GPU tests"
    )
    def test_prepared_operation_changed_payload_graph(self):
        import torch

        self.assertTrue(torch.version.hip)
        self.assertEqual(
            torch.cuda.get_device_properties(0).gcnArchName.split(":")[0], "gfx950"
        )
        lib = c.CDLL(os.environ["QVQ_GFX950_NATIVE_LIBRARY"])
        lib.qvq_gfx950_native_prepare.argtypes = (
            [c.POINTER(NativeConfig)]
            + [c.c_void_p] * 6
            + [c.c_size_t, c.c_void_p, c.c_size_t, c.c_void_p, c.POINTER(c.c_void_p)]
        )
        lib.qvq_gfx950_native_execute.argtypes = [c.c_void_p] * 7
        lib.qvq_gfx950_native_execute_capture.argtypes = [c.c_void_p] * 7
        lib.qvq_gfx950_native_prepare_runtime.argtypes = [
            c.POINTER(NativeConfig),
            c.c_void_p,
            c.c_size_t,
            c.c_void_p,
            c.c_size_t,
            c.c_void_p,
            c.POINTER(c.c_void_p),
        ]
        lib.qvq_gfx950_native_get_config.argtypes = [
            c.c_void_p,
            c.POINTER(NativeConfig),
        ]
        lib.qvq_gfx950_native_destroy.argtypes = [c.c_void_p]
        m, k, n = 8, 32, 48
        tiles = k * n // 256
        generator = torch.Generator().manual_seed(951)
        levels_cpu = (torch.arange(256, dtype=torch.float16) - 128) / 256
        for bits, alternate, bank_alt_id in (
            (4, 0x5A5A, 1),
            (5, 0x9696, 1),
            (6, 0x6969, 1),
            (7, 0xC3C3, 1),
            (8, 0, 0),
        ):
            config = NativeConfig(
                c.sizeof(NativeConfig),
                1,
                BlasConfig(c.sizeof(BlasConfig), 1, m, k, n, 1, 0, 0),
                bits,
                bank_alt_id,
                256,
                0,
            )
            x = torch.zeros((m, k), device="cuda", dtype=torch.float16)
            words = torch.zeros((tiles, 4 * bits), device="cuda", dtype=torch.int32)
            banks = torch.zeros(tiles, device="cuda", dtype=torch.uint8)
            levels = levels_cpu.cuda()
            y = torch.empty((m, n), device="cuda", dtype=torch.float32)
            scratch = torch.empty((n, k), device="cuda", dtype=torch.float16)
            workspace = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            plan = c.c_void_p()
            graph = None
            with torch.cuda.stream(stream):
                if bits >= 6:
                    y.fill_(123)
                    self.assertEqual(
                        lib.qvq_gfx950_native_prepare_runtime(
                            c.byref(config),
                            scratch.data_ptr(),
                            scratch.numel() * 2,
                            workspace.data_ptr(),
                            workspace.numel(),
                            stream.cuda_stream,
                            c.byref(plan),
                        ),
                        0,
                    )
                    self.assertTrue(
                        torch.equal(
                            y.cpu(), torch.full((m, n), 123, dtype=torch.float32)
                        )
                    )
                    # Validate this preparation route, then use its plan below.
                    prepared_runtime = plan
                    plan = c.c_void_p()
                else:
                    prepared_runtime = None
                self.assertEqual(
                    lib.qvq_gfx950_native_prepare(
                        c.byref(config),
                        x.data_ptr(),
                        words.data_ptr(),
                        levels.data_ptr(),
                        banks.data_ptr(),
                        y.data_ptr(),
                        scratch.data_ptr(),
                        scratch.numel() * 2,
                        workspace.data_ptr(),
                        workspace.numel(),
                        stream.cuda_stream,
                        c.byref(plan),
                    ),
                    0,
                )
                if prepared_runtime is not None:
                    self.assertEqual(lib.qvq_gfx950_native_destroy(plan), 0)
                    plan = prepared_runtime
                try:
                    expected_config = bytes(config)
                    config.bank_alt_id = 3
                    resolved = NativeConfig(c.sizeof(NativeConfig), 1)
                    self.assertEqual(
                        lib.qvq_gfx950_native_get_config(plan, c.byref(resolved)), 0
                    )
                    self.assertEqual(bytes(resolved), expected_config)
                    args = (
                        plan,
                        x.data_ptr(),
                        words.data_ptr(),
                        levels.data_ptr(),
                        banks.data_ptr(),
                        y.data_ptr(),
                    )
                    self.assertNotEqual(lib.qvq_gfx950_native_execute(*args, None), 0)
                    capture_stream = torch.cuda.Stream() if bits >= 6 else stream
                    self.assertNotEqual(
                        lib.qvq_gfx950_native_execute_capture(*args, capture_stream.cuda_stream), 0
                    )
                    if bits >= 6:
                        self.assertNotEqual(
                            lib.qvq_gfx950_native_execute(*args, capture_stream.cuda_stream), 0
                        )
                    stream.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    capture_call = (
                        lib.qvq_gfx950_native_execute_capture if bits >= 6
                        else lib.qvq_gfx950_native_execute
                    )
                    with torch.cuda.graph(graph, stream=capture_stream):
                        self.assertEqual(
                            capture_call(*args, capture_stream.cuda_stream), 0
                        )
                    for _ in range(3):
                        cpu_words = torch.randint(
                            0,
                            2**32,
                            (tiles, 4 * bits),
                            generator=generator,
                            dtype=torch.int64,
                        )
                        cpu_banks = torch.randint(
                            0, 256, (tiles,), generator=generator, dtype=torch.uint8
                        )
                        cpu_x = (
                            torch.randn(
                                (m, k), generator=generator, dtype=torch.float16
                            )
                            * 0.1
                        )
                        expected_w = torch.empty((n, k), dtype=torch.float64)
                        for tile, tile_words in enumerate(cpu_words.tolist()):
                            packed = sum(
                                word << (32 * i) for i, word in enumerate(tile_words)
                            )
                            circular = packed | (packed << (128 * bits))
                            for pair in range(128):
                                state = (circular >> ((127 - pair) * bits)) & 65535
                                state ^= (
                                    (int(cpu_banks[tile]) >> (pair // 16)) & 1
                                ) * alternate
                                mixed = ((state ^ (state >> 8)) * 40503 + 17011) & 65535
                                mixed ^= mixed >> 7
                                col, row = (
                                    (tile % (n // 16)) * 16 + (pair % 8) * 2,
                                    (tile // (n // 16)) * 16 + pair // 8,
                                )
                                expected_w[col, row] = levels_cpu[mixed >> 8]
                                expected_w[col + 1, row] = levels_cpu[mixed & 255]
                        words.copy_(cpu_words.to(torch.int32))
                        banks.copy_(cpu_banks)
                        x.copy_(cpu_x)
                        graph.replay()
                        stream.synchronize()
                        self.assertTrue(torch.equal(scratch.cpu().double(), expected_w))
                        error = (y.cpu().double() - cpu_x.double() @ expected_w.T).abs()
                        self.assertTrue(torch.isfinite(error).all())
                        self.assertLessEqual(error.mean().item(), 0.003)
                        self.assertLessEqual(error.max().item(), 0.006)
                        # Capturing on another stream must not change ordinary
                        # execution's binding or the frozen mathematical plan.
                        graph_y = y.clone()
                        self.assertEqual(
                            lib.qvq_gfx950_native_execute(*args, stream.cuda_stream), 0
                        )
                        stream.synchronize()
                        self.assertTrue(torch.equal(y, graph_y))
                finally:
                    stream.synchronize()
                    del graph
                    self.assertEqual(lib.qvq_gfx950_native_destroy(plan), 0)

    def test_configuration_validation_without_gpu_initialization(self):
        lib = c.CDLL(os.environ["QVQ_GFX950_NATIVE_LIBRARY"])
        query = lib.qvq_gfx950_native_scratch_bytes
        query.argtypes = [c.POINTER(NativeConfig), c.POINTER(c.c_size_t)]
        config = NativeConfig(
            c.sizeof(NativeConfig),
            1,
            BlasConfig(c.sizeof(BlasConfig), 1, 8, 5120, 1024, 1, 0, 0),
            5,
            1,
            256,
            0,
        )
        size = c.c_size_t()
        self.assertEqual(query(c.byref(config), c.byref(size)), 0)
        self.assertEqual(size.value, 5120 * 1024 * 2)
        for field, value in (
            ("version", 2),
            ("struct_size", 0),
            ("bank_alt_id", 4),
            ("decode_threads", 128),
            ("cache_policy", 1),
        ):
            invalid = NativeConfig.from_buffer_copy(config)
            setattr(invalid, field, value)
            size.value = 123
            self.assertNotEqual(query(c.byref(invalid), c.byref(size)), 0)
            self.assertEqual(size.value, 0)
        w4 = NativeConfig.from_buffer_copy(config)
        w4.transition_bits = 8
        w4.bank_alt_id = 0
        self.assertEqual(query(c.byref(w4), c.byref(size)), 0)
        self.assertEqual(size.value, 5120 * 1024 * 2)
        for field, value in (
            ("e", 2),
            ("m", 0),
            ("k", 17),
            ("n", -1),
            ("solution_index", -1),
            ("reserved", 1),
            ("k", 1048576),
        ):
            invalid = NativeConfig.from_buffer_copy(config)
            setattr(invalid.gemm, field, value)
            self.assertNotEqual(query(c.byref(invalid), c.byref(size)), 0)
        self.assertNotEqual(query(None, c.byref(size)), 0)
        self.assertNotEqual(query(c.byref(config), None), 0)

    @unittest.skipUnless(
        os.environ.get("QVQ_GFX950_NATIVE_GPU_TEST"), "opt in to gfx950 GPU tests"
    )
    def test_standalone_rocblas_autotune_returns_frozen_solution(self):
        import torch

        lib = c.CDLL(os.environ["QVQ_GFX950_NATIVE_LIBRARY"])
        lib.qvq_gfx950_rocblas_prepare.argtypes = (
            [c.c_int] * 3
            + [c.c_void_p] * 2
            + [c.c_size_t, c.POINTER(c.c_void_p)]
        )
        lib.qvq_gfx950_rocblas_execute.argtypes = [c.c_void_p] * 5
        lib.qvq_gfx950_rocblas_destroy.argtypes = [c.c_void_p]

        class Options(c.Structure):
            _fields_ = [
                ("struct_size", c.c_uint32),
                ("version", c.c_uint32),
                ("warmup_iterations", c.c_uint32),
                ("benchmark_iterations", c.c_uint32),
                ("reserved", c.c_uint32),
            ]

        class Result(c.Structure):
            _fields_ = [
                ("struct_size", c.c_uint32),
                ("version", c.c_uint32),
                ("solution_index", c.c_int32),
                ("candidates_tested", c.c_uint32),
                ("candidates_failed", c.c_uint32),
                ("samples", c.c_uint32),
                ("median_us", c.c_float),
                ("reserved", c.c_uint32),
            ]

        lib.qvq_gfx950_rocblas_autotune.argtypes = (
            [c.c_void_p] * 4 + [c.POINTER(Options), c.POINTER(Result)]
        )
        m, k, n = 1, 256, 256
        gen = torch.Generator(device="cuda").manual_seed(952)
        x = torch.randn((m, k), generator=gen, device="cuda", dtype=torch.float16)
        weights = torch.randn((n, k), generator=gen, device="cuda", dtype=torch.float16)
        y = torch.empty((m, n), device="cuda", dtype=torch.float32)
        workspace = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
        stream = torch.cuda.Stream()
        plan = c.c_void_p()
        with torch.cuda.stream(stream):
            self.assertEqual(
                lib.qvq_gfx950_rocblas_prepare(
                    m,
                    k,
                    n,
                    stream.cuda_stream,
                    workspace.data_ptr(),
                    workspace.numel(),
                    c.byref(plan),
                ),
                0,
            )
            try:
                options = Options(c.sizeof(Options), 1, 0, 1, 0)
                result = Result(c.sizeof(Result), 1)
                self.assertEqual(
                    lib.qvq_gfx950_rocblas_autotune(
                        plan,
                        x.data_ptr(),
                        weights.data_ptr(),
                        y.data_ptr(),
                        c.byref(options),
                        c.byref(result),
                    ),
                    0,
                )
                # The standard rocBLAS algorithm (solution 0) is a valid
                # measured winner; zero is not an autotune failure sentinel.
                self.assertGreaterEqual(result.solution_index, 0)
                self.assertGreater(result.candidates_tested, 0)
                self.assertEqual(result.samples, result.candidates_tested)
                self.assertTrue(result.median_us > 0)
                self.assertEqual(
                    lib.qvq_gfx950_rocblas_execute(
                        plan,
                        x.data_ptr(),
                        weights.data_ptr(),
                        y.data_ptr(),
                        stream.cuda_stream,
                    ),
                    0,
                )
                stream.synchronize()
                error = (y.double() - x.double() @ weights.double().T).abs()
                self.assertLessEqual(error.mean().item(), 0.003)
                self.assertLessEqual(error.max().item(), 0.006)
            finally:
                stream.synchronize()
                self.assertEqual(lib.qvq_gfx950_rocblas_destroy(plan), 0)


if __name__ == "__main__":
    unittest.main()
