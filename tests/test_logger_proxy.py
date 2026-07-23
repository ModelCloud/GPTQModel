# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

# -- do not touch
# GPU=-1
import os


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# -- end do not touch

import copy  # noqa: E402
import unittest  # noqa: E402

from gptqmodel.utils.logger import _AdaptiveLoggerProxy  # noqa: E402


class _StubLogger:
    def info(self, *args, **kwargs):
        return ("info", args, kwargs)


class TestAdaptiveLoggerProxy(unittest.TestCase):
    def test_forwards_attributes(self):
        proxy = _AdaptiveLoggerProxy(_StubLogger())
        self.assertEqual(proxy.info(1, k=2), ("info", (1,), {"k": 2}))

    def test_deepcopy_does_not_recurse(self):
        # Deepcopying an object that holds the proxy reconstructs the proxy via
        # __new__ (without __init__); before the __getattr__ guard this looped
        # through self._logger forever and raised RecursionError.
        holder = {"log": _AdaptiveLoggerProxy(_StubLogger())}
        clone = copy.deepcopy(holder)
        self.assertEqual(clone["log"].info(3), ("info", (3,), {}))

    def test_missing_logger_raises_attributeerror(self):
        # A proxy with no _logger set (e.g. mid-reconstruction) must surface a
        # clean AttributeError rather than recursing.
        proxy = _AdaptiveLoggerProxy.__new__(_AdaptiveLoggerProxy)
        with self.assertRaises(AttributeError):
            _ = proxy.info


if __name__ == "__main__":
    unittest.main()
