"""Host-only rejection checks for the external HIP ABI (no GPU initialization)."""

import ctypes as c
import os
import unittest


class Spec(c.Structure):
    _fields_ = [
        (name, c.c_uint32)
        for name in (
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
    ]


@unittest.skipUnless(
    os.environ.get("QVQ_GFX950_TEST_LIBRARY"), "requires explicit native library"
)
class NativeValidationTests(unittest.TestCase):
    def setUp(self):
        self.lib = c.CDLL(os.environ["QVQ_GFX950_TEST_LIBRARY"])
        self.lib.qvq_gfx950_last_error.restype = c.c_char_p
        self.lib.qvq_gfx950_prepare.argtypes = [
            c.POINTER(Spec),
            c.c_void_p,
            c.c_size_t,
            c.c_char_p,
            c.c_int,
            c.c_void_p,
            c.POINTER(c.c_void_p),
        ]
        self.lib.qvq_gfx950_execute.argtypes = [c.c_void_p] * 7
        self.lib.qvq_gfx950_destroy.argtypes = [c.c_void_p]

    def test_invalid_descriptors_rejected_before_module_loading(self):
        # Dummy bytes must never reach HIP: every descriptor below is invalid.
        image = c.create_string_buffer(64)
        for field, value, message in (
            ("abi_version", 0, "incompatible"),
            ("operation_version", 2, "incompatible"),
            ("m", 0, "geometry"),
            ("k", 255, "geometry"),
            ("n", 255, "geometry"),
            ("transition_bits", 3, "geometry"),
            ("bank_alt_id", 0, "geometry"),
            ("grid_x", 0, "geometry"),
            ("threads", 65, "geometry"),
            ("threads", 1088, "geometry"),
        ):
            with self.subTest(field=field, value=value):
                spec = Spec(1, 1, 1, 256, 256, 6, 3, 4, 512, 256)
                setattr(spec, field, value)
                result = c.c_void_p(123)
                status = self.lib.qvq_gfx950_prepare(
                    c.byref(spec),
                    image,
                    64,
                    b"unused",
                    0,
                    None,
                    c.byref(result),
                )
                self.assertNotEqual(status, 0)
                self.assertIsNone(result.value)
                self.assertIn(message, self.lib.qvq_gfx950_last_error().decode())

    def test_null_execution_rejected(self):
        self.assertNotEqual(self.lib.qvq_gfx950_execute(*([None] * 7)), 0)
        self.assertIn(b"null", self.lib.qvq_gfx950_last_error())

    def test_null_destroy_is_noop(self):
        self.assertEqual(self.lib.qvq_gfx950_destroy(None), 0)


if __name__ == "__main__":
    unittest.main()
