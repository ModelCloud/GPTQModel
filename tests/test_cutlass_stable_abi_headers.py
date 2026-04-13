# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]


class CutlassStableAbiHeaderTests(unittest.TestCase):
    def test_common_header_uses_stable_abi_shim_check(self) -> None:
        header = (
            _REPO_ROOT / "gptqmodel_ext" / "cutlass_extensions" / "common.hpp"
        ).read_text(encoding="utf-8")

        self.assertIn("#include <torch/headeronly/util/shim_utils.h>", header)
        self.assertIn(
            "STD_TORCH_CHECK(error == cutlass::Status::kSuccess,",
            header,
        )

    def test_torch_utils_header_supports_stable_and_unstable_torch_abi(self) -> None:
        header = (
            _REPO_ROOT / "gptqmodel_ext" / "cutlass_extensions" / "torch_utils.hpp"
        ).read_text(encoding="utf-8")

        self.assertIn("shared between _C (unstable ABI, used by machete)", header)
        self.assertIn("#ifdef TORCH_TARGET_VERSION", header)
        self.assertIn("using TorchTensor = torch::stable::Tensor;", header)
        self.assertIn("using TorchTensor = torch::Tensor;", header)
        self.assertIn("#define TORCH_UTILS_CHECK STD_TORCH_CHECK", header)
        self.assertIn("#define TORCH_UTILS_CHECK TORCH_CHECK", header)
        self.assertIn("static inline auto make_cute_layout(TorchTensor const& tensor,", header)
        self.assertIn("std::optional<TorchTensor> const& tensor,", header)
        self.assertIn("struct equivalent_cutlass_type<TorchHalf>", header)
        self.assertIn("struct equivalent_cutlass_type<TorchBFloat16>", header)
        self.assertIn("static inline constexpr TorchScalarType equivalent_scalar_type_v =", header)


if __name__ == "__main__":
    unittest.main()
