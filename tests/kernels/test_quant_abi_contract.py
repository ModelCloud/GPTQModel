"""Run with QVQ_QUANT_ABI_LIBRARY=/absolute/path/libqvq_quant.so.

These are ABI/rejection tests, not target-kernel numerical validation.
"""
import ctypes as C
import os
import subprocess
import sys
import unittest

import torch


class Tensor(C.Structure):
    _fields_ = [("data", C.c_void_p), ("bytes", C.c_uint64),
                ("shape", C.c_int64 * 4), ("rank", C.c_uint32), ("dtype", C.c_uint32)]


class Config(C.Structure):
    _fields_ = [(x, C.c_uint32) for x in ("struct_bytes", "abi_version", "operation")]
    _fields_ += [("device", C.c_int32)]
    _fields_ += [(x, C.c_int64) for x in ("weight_type", "group_size", "k", "n")]
    _fields_ += [(x, C.c_uint32) for x in ("activation_dtype", "output_dtype", "num_bits", "transpose")]
    _fields_ += [(x, C.c_int32) for x in ("mode", "m_tiles", "split_k", "ctas", "cta_quad", "threads", "stages")]
    _fields_ += [("tile_n", C.c_int32), ("chunk_m", C.c_int32), ("reserved", C.c_uint32), ("schedule", C.c_char_p)]


@unittest.skipUnless(os.getenv("QVQ_QUANT_ABI_LIBRARY"), "native ABI build required")
class Contract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lib = C.CDLL(os.environ["QVQ_QUANT_ABI_LIBRARY"])
        cls.lib.qvq_quant_prepare.argtypes = [C.POINTER(Config), C.POINTER(Tensor), C.c_uint32,
            Tensor, C.c_void_p, C.POINTER(C.c_void_p), C.c_char_p, C.c_uint64]
        cls.lib.qvq_quant_launch.argtypes = [C.c_void_p, C.c_void_p, C.c_char_p, C.c_uint64]
        cls.lib.qvq_quant_destroy.argtypes = [C.c_void_p, C.c_char_p, C.c_uint64]
        cls.lib.qvq_machete_schedules.argtypes = [C.c_uint32, C.c_int64, C.POINTER(C.c_uint32),
            C.c_char_p, C.c_uint64, C.POINTER(C.c_uint64), C.c_char_p, C.c_uint64]

    def setUp(self):
        self.error = C.create_string_buffer(1024)
        self.config = Config(struct_bytes=C.sizeof(Config), abi_version=1, operation=1,
                             schedule=b"explicit-test-schedule")

    def prepare_error(self, stream=1):
        handle = C.c_void_p(123)
        code = self.lib.qvq_quant_prepare(C.byref(self.config), (Tensor * 6)(), 6,
            Tensor(), stream, C.byref(handle), self.error, len(self.error))
        self.assertEqual(code, -1)
        self.assertIsNone(handle.value)
        return self.error.value.decode()

    def test_version_and_layout(self):
        self.assertEqual(self.lib.qvq_quant_abi_version(), 1)
        self.assertEqual(C.sizeof(Config), 112)
        self.assertEqual(C.sizeof(Tensor), 56)

    def test_rejected_headers(self):
        self.config.abi_version = 2
        self.assertIn("header", self.prepare_error())
        self.config.abi_version = 1
        self.config.struct_bytes = 8
        self.assertIn("header", self.prepare_error())

    def test_no_implicit_machete_schedule(self):
        self.config.schedule = None
        self.assertIn("explicit compiled schedule", self.prepare_error())

    def test_default_stream_rejected(self):
        self.assertIn("nondefault stream", self.prepare_error(None))

    def test_unused_tuning_rejected(self):
        self.config.ctas = 1
        self.assertIn("decode tuning", self.prepare_error())

    def test_null_lifecycle(self):
        self.assertEqual(self.lib.qvq_quant_destroy(None, self.error, 1024), 0)
        self.assertEqual(self.lib.qvq_quant_launch(None, None, self.error, 1024), -1)

    def test_schedule_query_missing_operator_or_exact_results(self):
        # A real loaded operator is allowed; otherwise verify exceptions stay
        # within the C boundary. No fake kernel is installed under a real name.
        required = C.c_uint64()
        ds = (C.c_uint32 * 5)(1, 0, 0, 0, 1)
        result = self.lib.qvq_machete_schedules(1, 0, ds, None, 0,
            C.byref(required), self.error, 1024)
        if result == 0:
            self.assertGreaterEqual(required.value, 1)
        else:
            self.assertTrue(self.error.value)

    def test_schedule_query_transport(self):
        # Isolated metadata-only mock, never used as numerical kernel evidence.
        code = r'''
import ctypes as C
import os
import torch
lib = torch.library.Library("gptqmodel_machete", "DEF")
lib.define("machete_supported_schedules(ScalarType a, int b, ScalarType? gs, ScalarType? gz, ScalarType? cs, ScalarType? ts, ScalarType? out) -> str[]")
lib.impl("machete_supported_schedules", lambda *args: ["tile128", "tile256"], "CompositeExplicitAutograd")
native = C.CDLL(os.environ["QVQ_QUANT_ABI_LIBRARY"])
native.qvq_machete_schedules.argtypes = [C.c_uint32, C.c_int64, C.POINTER(C.c_uint32), C.c_char_p, C.c_uint64, C.POINTER(C.c_uint64), C.c_char_p, C.c_uint64]
ds, size, error = (C.c_uint32*5)(), C.c_uint64(), C.create_string_buffer(1024)
def query(buf, capacity):
    return native.qvq_machete_schedules(1, 0, ds, buf, capacity, C.byref(size), error, 1024)
assert query(None, 0) == 0, error.value
assert size.value == len(b"tile128\ntile256\n") + 1
buf = C.create_string_buffer(size.value)
assert query(buf, size.value - 1) == -1
assert query(buf, size.value) == 0, error.value
assert buf.value == b"tile128\ntile256\n"
'''
        subprocess.run([sys.executable, "-c", code], check=True, timeout=60)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA needed for architecture rejection")
    def test_wrong_architecture(self):
        if torch.cuda.get_device_capability() != (8, 0):
            self.skipTest("test specifically covers available SM80 host")
        stream = torch.cuda.Stream()
        for operation in (1, 3, 4, 5, 6):
            self.config.operation = operation
            self.config.schedule = b"explicit" if operation == 1 else None
            self.assertIn("requires SM", self.prepare_error(stream.cuda_stream))


if __name__ == "__main__":
    unittest.main()
