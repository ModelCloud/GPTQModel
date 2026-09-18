# SPDX-License-Identifier: Apache-2.0
"""CPU checks for dispatch forwarding and torch.ops mutation declarations."""
import ast
import re
from pathlib import Path

import pytest
import torch

from gptqmodel.utils import marlin as marlin_utils
from gptqmodel.utils.marlin_scalar_type import scalar_types


@pytest.mark.parametrize("dtype,suffix", [(torch.float16, "fp16"), (torch.bfloat16, "bf16")])
@pytest.mark.parametrize("provided", ["both", "reduction", "permutation"])
def test_explicit_scratch_reaches_dtype_operation(monkeypatch, dtype, suffix, provided):
    calls = []
    c_tmp = torch.empty(1024, dtype=torch.float32) if provided != "permutation" else None
    a_tmp = torch.empty(128, dtype=dtype) if provided != "reduction" else None

    def resolve(*, dtype, op_name):
        assert op_name == "gptq_marlin_gemm_" + suffix
        def op(*args):
            calls.append(args)
            return torch.empty(args[11], args[12], dtype=dtype)
        return op

    monkeypatch.setattr(marlin_utils, "_marlin_resolve_op", resolve)
    out = marlin_utils.gptq_marlin_gemm(
        torch.ones(1, 128, dtype=dtype), None, torch.zeros(8, 128, dtype=torch.int32),
        None, torch.ones(1, 64, dtype=dtype), None, None, None, None,
        torch.zeros(1, dtype=torch.int32), scalar_types.uint4b8, 1, 64, 128,
        c_tmp=c_tmp, a_tmp=a_tmp)
    assert out.shape == (1, 64)
    assert calls[0][-2] is c_tmp
    assert calls[0][-1] is a_tmp
    assert calls[0][10:14] == (scalar_types.uint4b8.id, 1, 64, 128)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_capacity_query_forwards_executed_dimensions(monkeypatch, dtype):
    a = torch.empty(17, 320, dtype=dtype)
    def resolve(*, dtype, op_name):
        assert dtype == a.dtype
        assert op_name == "marlin_scratch_sizes"
        def query(*args):
            assert args[0] is a
            assert args[1:] == (17, 320, True, True)
            return 1234, 5440
        return query
    monkeypatch.setattr(marlin_utils, "_marlin_resolve_op", resolve)
    assert marlin_utils.marlin_scratch_sizes(a, 17, 320, True, True) == (1234, 5440)


def schemas(path):
    source = path.read_text()
    result = []
    for match in re.finditer(r'm\.def\(\s*((?:"[^"\n]*"\s*)+)\);', source):
        text = "".join(ast.literal_eval(s) for s in re.findall(r'"[^"\n]*"', match.group(1)))
        result.append(torch._C.parse_schema(text))
    return result


@pytest.mark.parametrize("suffix", ["fp16", "bf16"])
def test_mutable_schema_and_capacity_contract(suffix):
    root = Path(__file__).resolve().parents[1] / "gptqmodel_ext/marlin"
    parsed = schemas(root / f"marlin_torch_{suffix}.cpp")
    gemm = next(s for s in parsed if s.name == f"gptq_marlin_gemm_{suffix}")
    by_name = {arg.name: arg for arg in gemm.arguments}
    for name in ("c", "workspace", "c_tmp", "a_tmp"):
        assert by_name[name].alias_info.is_write, name
    for name in ("a", "b_q_weight", "b_scales", "g_idx", "perm"):
        assert by_name[name].alias_info is None or not by_name[name].alias_info.is_write
    assert by_name["c_tmp"].has_default_value() and by_name["c_tmp"].default_value is None
    assert by_name["a_tmp"].has_default_value() and by_name["a_tmp"].default_value is None
    assert gemm.returns[0].alias_info.before_set == by_name["c"].alias_info.before_set
    capacity = next(s for s in parsed if s.name == "marlin_scratch_sizes")
    assert [arg.name for arg in capacity.arguments] == [
        "a", "size_m", "size_k", "use_fp32_reduce", "has_act_order"]
    assert [str(ret.type) for ret in capacity.returns] == ["int", "int"]
