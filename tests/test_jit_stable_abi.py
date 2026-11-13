# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Source-level checks for JIT kernels migrated to the torch stable ABI.

The stable ABI ("AnyTorch") lets one compiled artifact run on every newer
torch release. These tests assert that a migrated loader declares the ABI
floor and that its source graph only includes stable torch headers, without
requiring a GPU or a compiler.
"""

from __future__ import annotations

from gptqmodel.utils import cpp
from gptqmodel.utils import awq
from gptqmodel.utils import exllamav2
from gptqmodel.utils import hadamard
from gptqmodel.utils import marlin
from gptqmodel.utils import paroquant
from gptqmodel.utils import qqq


def _assert_stable(loader) -> None:
    assert loader.torch_stable_abi_target == cpp._TORCH_STABLE_ABI_TARGET
    sources = loader._resolve_sequence(loader.sources)
    include_paths = loader._resolved_extra_include_paths()
    python_dependent, stable, non_stable_headers = loader._source_abi_details(
        sources, include_paths
    )
    assert stable, non_stable_headers
    assert not python_dependent
    assert non_stable_headers == ()
    if loader.requires_cuda:
        assert "-DUSE_CUDA" in loader._resolved_extra_cflags()


def test_stable_abi_floor_is_torch_2_10():
    assert cpp._TORCH_STABLE_ABI_TARGET == (2, 10)
    assert (
        cpp.torch_stable_abi_target_define(*cpp._TORCH_STABLE_ABI_TARGET)
        == "-DTORCH_TARGET_VERSION=0x020A000000000000"
    )


def test_pack_block_cpu_uses_stable_abi():
    _assert_stable(cpp._pack_block_extension())


def test_floatx_cpu_uses_stable_abi():
    _assert_stable(cpp._floatx_cpu_extension())


def test_hadamard_uses_stable_abi():
    _assert_stable(hadamard._HADAMARD_TORCH_OPS_EXTENSION)


def test_qqq_uses_stable_abi():
    _assert_stable(qqq._QQQ_TORCH_OPS_EXTENSION)


def test_awq_uses_stable_abi():
    _assert_stable(awq._AWQ_TORCH_OPS_EXTENSION)


def test_paroquant_uses_stable_abi():
    _assert_stable(paroquant._PAROQUANT_ROTATION_EXTENSION)


def test_marlin_fp16_uses_stable_abi():
    _assert_stable(marlin._MARLIN_FP16_TORCH_OPS_EXTENSION)


def test_marlin_bf16_uses_stable_abi():
    _assert_stable(marlin._MARLIN_BF16_TORCH_OPS_EXTENSION)


def test_exllamav2_gptq_uses_stable_abi():
    _assert_stable(exllamav2._EXLLAMAV2_GPTQ_TORCH_OPS_EXTENSION)


def test_exllamav2_awq_uses_stable_abi():
    _assert_stable(exllamav2._EXLLAMAV2_AWQ_TORCH_OPS_EXTENSION)
