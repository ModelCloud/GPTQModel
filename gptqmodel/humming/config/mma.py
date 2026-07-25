# Copyright 2024-2025 InclusionAI (https://github.com/inclusionAI) and the Humming team.
# SPDX-License-Identifier: Apache-2.0
#
# This file is vendored from https://github.com/inclusionAI/humming and used under
# the terms of the Apache License, Version 2.0.
# See gptqmodel/humming/LICENSE for the full license text.

import math
import re

from .. import dtypes as dtypes
from .enum import MmaType

DTYPE_BIT_WIDTH_MAP = {
    "f32": 32,
    "s32": 32,
    "f16": 16,
    "bf16": 16,
    "e4m3": 8,
    "e5m2": 8,
    "e8m0": 8,
    "s8": 8,
    "e3m2": 6,
    "e2m3": 6,
    "e2m1": 4,
    "s4": 4,
}

DTYPE_MAP = {
    dtypes.float32: "f32",
    dtypes.int32: "s32",
    dtypes.float16: "f16",
    dtypes.bfloat16: "bf16",
    dtypes.float8e4m3: "e4m3",
    dtypes.float8e5m2: "e5m2",
    dtypes.float8e8m0: "e8m0",
    dtypes.int8: "s8",
    dtypes.float6e3m2: "e3m2",
    dtypes.float6e2m3: "e2m3",
    dtypes.float4e2m1: "e2m1",
    dtypes.int4: "s4",
    # Hardware supports these dtypes (e3m4/e0m3) but exposes no matching PTX types.
    # We substitute other PTX types and patch the cubin afterwards.
    dtypes.float8e3m4: "e5m2",
    dtypes.float4e0m3: "e2m1",
}

SF_DTYPE_MAP = {
    "e8m0": "ue8m0",
    "e4m3": "ue4m3",
}


def calc_reg_count(rows, cols, ptx_dtype):
    total_bits = rows * cols * DTYPE_BIT_WIDTH_MAP[ptx_dtype]
    assert total_bits % (32 * 32) == 0
    reg_count = total_bits // (32 * 32)
    return reg_count


class MmaOpClassImpl:
    def __init__(self, m, n, k, a_dtype, b_dtype, cd_dtype):
        self.shape = (m, n, k)
        self.a_dtype = a_dtype if isinstance(a_dtype, str) else DTYPE_MAP[a_dtype]
        self.b_dtype = b_dtype if isinstance(b_dtype, str) else DTYPE_MAP[b_dtype]
        self.cd_dtype = cd_dtype if isinstance(cd_dtype, str) else DTYPE_MAP[cd_dtype]

        f6f4_types = ("e3m2", "e2m3", "e2m1")
        self.use_f8f6f4 = self.a_dtype in ("e4m3", "e5m2") and self.b_dtype in f6f4_types
        self.native_mixed = self.use_f8f6f4

        b_reg_dtype = "e4m3" if self.use_f8f6f4 else self.b_dtype
        self.reg_a_count = calc_reg_count(m, k, self.a_dtype)
        self.reg_b_count = calc_reg_count(k, n, b_reg_dtype)
        self.reg_cd_count = calc_reg_count(m, n, self.cd_dtype)
        if self.cd_dtype == "f16":
            self.val_type_cd = "half"
            self.reg_cd_type = "uint32_t"
        elif self.cd_dtype == "bf16":
            self.val_type_cd = "nv_bfloat16"
            self.reg_cd_type = "uint32_t"
        elif self.cd_dtype == "f32":
            self.val_type_cd = "float"
            self.reg_cd_type = "float"
        elif self.cd_dtype == "s32":
            self.val_type_cd = "int32_t"
            self.reg_cd_type = "uint32_t"
        else:
            raise ValueError(f"Invalid cd_dtype: {cd_dtype}")

    def to_cpp_str(self, include_class_name=False):
        reg_cd_type = self.reg_cd_type
        lines = [
            "static constexpr MmaType kMmaType = MmaType::MMA;",
            f"using MmaShape = Shape<{self.shape[0]}, {self.shape[1]}, {self.shape[2]}>;",
            "",
            f"using ValTypeC = {self.val_type_cd};",
            f"using ValTypeD = {self.val_type_cd};",
            "",
            f"static constexpr uint32_t kATypeBits = {DTYPE_BIT_WIDTH_MAP[self.a_dtype]};",
            f"static constexpr uint32_t kBTypeBits = {DTYPE_BIT_WIDTH_MAP[self.b_dtype]};",
            f"static constexpr uint32_t kCTypeBits = {DTYPE_BIT_WIDTH_MAP[self.cd_dtype]};",
            f"static constexpr uint32_t kDTypeBits = {DTYPE_BIT_WIDTH_MAP[self.cd_dtype]};",
            f"static constexpr bool kNativeMixed = {'true' if self.native_mixed else 'false'};",
            "",
            f"using ARegisters = uint32_t[{self.reg_a_count}];",
            f"using BRegisters = uint32_t[{self.reg_b_count}];",
            f"using CRegisters = {self.reg_cd_type}[{self.reg_cd_count}];",
            f"using DRegisters = {self.reg_cd_type}[{self.reg_cd_count}];",
            "",
            "CUDA_INLINE",
            f"static void fma(uint32_t *a, uint32_t *b, {reg_cd_type} *c, {reg_cd_type} *d) {{",
            *self.generate_ptx(indent=2).strip("\n").split("\n"),
            "};",
        ]

        code = "\n".join("  " + x if x else x for x in lines)
        if include_class_name:
            code = f"class MmaOpClass {{\n{code}\n}};"

        return code

    def generate_ptx(self, indent=0):
        a_dtype = self.a_dtype
        b_dtype = self.b_dtype
        cd_dtype = self.cd_dtype
        shape = self.shape

        asm_op = "mma.sync.aligned"
        if self.use_f8f6f4:
            asm_op += ".kind::f8f6f4"
        asm_op += f".m{shape[0]}n{shape[1]}k{shape[2]}.row.col"
        asm_op += f".{cd_dtype}.{a_dtype}.{b_dtype}.{cd_dtype}"
        if "s" in a_dtype:
            asm_op += ".satfinite"

        start = 0
        end = 0
        param_placeholders_list = []
        counts = [self.reg_cd_count, self.reg_a_count, self.reg_b_count, self.reg_cd_count]
        for i in range(len(counts)):
            end += counts[i]
            placeholder_str = ", ".join(f"%{x}" for x in range(start, end))
            param_placeholders_list.append("{" + placeholder_str + "}")
            start += counts[i]

        a_params = []
        b_params = []
        c_params = []
        d_params = []
        for i in range(self.reg_a_count):
            a_params.append(f' "r"(a[{i}])')
        for i in range(self.reg_b_count):
            b_params.append(f' "r"(b[{i}])')
        for i in range(self.reg_cd_count):
            t = "f" if cd_dtype == "f32" else "r"
            c_params.append(f' "{t}"(c[{i}])')
            d_params.append(f'"+{t}"(d[{i}])')

        asm_code = f"""
        asm volatile(
          "{asm_op} "
          "{", ".join(param_placeholders_list)};\\n"
          : {", ".join(d_params)}
          : {", ".join(a_params)},
            {", ".join(b_params)},
            {", ".join(c_params)}
        );
        """

        space_count = len(re.findall("^\n( +)", asm_code)[0])
        asm_code = asm_code.replace("\n" + " " * space_count, "\n").strip()
        asm_code = "".join("\n" + " " * indent + x for x in asm_code.split("\n"))

        return asm_code


class WgmmaOpClassImpl:
    def __init__(self, m, n, k, a_dtype, b_dtype, cd_dtype):
        self.shape = (m, n, k)
        self.a_dtype = a_dtype if isinstance(a_dtype, str) else DTYPE_MAP[a_dtype]
        self.b_dtype = b_dtype if isinstance(b_dtype, str) else DTYPE_MAP[b_dtype]
        self.cd_dtype = cd_dtype if isinstance(cd_dtype, str) else DTYPE_MAP[cd_dtype]

        self.reg_b_count = calc_reg_count(n, k, self.b_dtype) // 4
        self.reg_cd_count = calc_reg_count(m, n, self.cd_dtype) // 4
        if self.cd_dtype == "f16":
            self.val_type_cd = "half"
            self.reg_cd_type = "uint32_t"
        elif self.cd_dtype == "bf16":
            self.val_type_cd = "nv_bfloat16"
            self.reg_cd_type = "uint32_t"
        elif self.cd_dtype == "f32":
            self.val_type_cd = "float"
            self.reg_cd_type = "float"
        elif self.cd_dtype == "s32":
            self.val_type_cd = "int32_t"
            self.reg_cd_type = "uint32_t"
        else:
            raise ValueError(f"Invalid cd_dtype: {cd_dtype}")

    def to_cpp_str(self, include_class_name=False):
        reg_cd_type = self.reg_cd_type
        lines = [
            "static constexpr MmaType kMmaType = MmaType::WGMMA;",
            f"using MmaShape = Shape<{self.shape[0]}, {self.shape[1]}, {self.shape[2]}>;",
            "",
            f"using ValTypeC = {self.val_type_cd};",
            f"using ValTypeD = {self.val_type_cd};",
            "",
            f"static constexpr uint32_t kATypeBits = {DTYPE_BIT_WIDTH_MAP[self.a_dtype]};",
            f"static constexpr uint32_t kBTypeBits = {DTYPE_BIT_WIDTH_MAP[self.b_dtype]};",
            f"static constexpr uint32_t kCTypeBits = {DTYPE_BIT_WIDTH_MAP[self.cd_dtype]};",
            f"static constexpr uint32_t kDTypeBits = {DTYPE_BIT_WIDTH_MAP[self.cd_dtype]};",
            "static constexpr bool kNativeMixed = false;",
            "",
            f"using BRegisters = uint32_t[{self.reg_b_count}];",
            f"using CRegisters = {self.reg_cd_type}[{self.reg_cd_count}];",
            f"using DRegisters = {self.reg_cd_type}[{self.reg_cd_count}];",
            "",
            "CUDA_INLINE",
            f"static void fma(uint64_t &desc, uint32_t *b, {reg_cd_type} *d, bool pred = true) {{",
            *self.generate_ptx(indent=2, has_scale_d=True).strip("\n").split("\n"),
            "};",
        ]

        code = "\n".join("  " + x if x else x for x in lines)
        if include_class_name:
            code = f"class MmaOpClass {{\n{code}\n}};"

        return code

    def generate_ptx(self, indent=2, has_scale_d=True):
        a_dtype = self.a_dtype
        b_dtype = self.b_dtype
        cd_dtype = self.cd_dtype
        m, n, k = self.shape

        # Swap M<->N and A-dtype<->B-dtype in PTX: project's A becomes wgmma's B and
        # project's B becomes wgmma's A. The PTX dtype suffix order is .cd.a.b, so
        # the wgmma A slot takes project's b_dtype and the wgmma B slot takes a_dtype.
        asm_op = f"wgmma.mma_async.sync.aligned.m{n}n{m}k{k}"
        asm_op += f".{cd_dtype}.{b_dtype}.{a_dtype}"
        if "s" in b_dtype:
            asm_op += ".satfinite"

        start = 0
        end = 0
        param_placeholders_list = []
        counts = [self.reg_cd_count, self.reg_b_count]
        for i in range(len(counts)):
            end += counts[i]
            placeholder_str = ", ".join(f"%{x}" for x in range(start, end))
            param_placeholders_list.append("{" + placeholder_str + "}")
            start += counts[i]
        param_placeholders_list.append(f"%{sum(counts)}")

        other_ptx_args = ", p" if has_scale_d else ", 1"
        # The dtype-specific PTX tail args (scale/trans flags) gate on the wgmma-A
        # operand dtype, which after the swap is project's b_dtype.
        if self.b_dtype in ["f16", "bf16"]:
            other_ptx_args += ", 1, 1, 0"
        elif self.b_dtype in ["e4m3", "e5m2", "e2m1"]:
            other_ptx_args += ", 1, 1"

        # Project A's smem descriptor fills the wgmma B operand.
        a_desc_param = ' "l"(desc)'
        # Project B's registers fill the wgmma A operand.
        b_params = []
        cd_params = []
        for i in range(self.reg_b_count):
            b_params.append(f' "r"(b[{i}])')
        for i in range(self.reg_cd_count):
            t = "f" if cd_dtype == "f32" else "r"
            cd_params.append(f'"+{t}"(d[{i}])')

        cd_param_str = ""
        for i in range(math.ceil(len(cd_params) / 4)):
            cd_params_part = cd_params[i * 4 : (i + 1) * 4]
            cd_params_part_str = ", ".join(cd_params_part) + ",\n"
            if cd_param_str:
                cd_params_part_str = "    " + cd_params_part_str

            cd_param_str += cd_params_part_str

        cd_param_str = cd_param_str.strip().strip(",")

        if has_scale_d:
            asm_code = f"""
            asm volatile(
              "{{\\n"
                ".reg .pred p;\\n"
                "setp.ne.b32 p, %{sum(counts) + 1}, 0;\\n"
                "{asm_op} "
                "{", ".join(param_placeholders_list)}{other_ptx_args};\\n"
              "}}\\n"
              : {cd_param_str}
              : {", ".join(b_params)},
                {a_desc_param}, "r"((uint32_t)pred)
            );
            """
        else:
            asm_code = f"""
            asm volatile(
            "{asm_op} "
            "{", ".join(param_placeholders_list)}{other_ptx_args};\\n"
            : {cd_param_str}
            : {", ".join(b_params)},
                {a_desc_param}
            );
            """

        space_count = len(re.findall("^\n( +)", asm_code)[0])
        asm_code = asm_code.replace("\n" + " " * space_count, "\n").strip()
        asm_code = "".join("\n" + " " * indent + x for x in asm_code.split("\n"))

        return asm_code


class MxMmaOpClassImpl:
    """Microscale (block-scaled) warp-level ``mma.sync`` for SM120.

    Emits ``mma.sync.aligned.<kind>.block_scale.<scale_vec>...`` which consumes a
    per-block scale factor for each of A and B in addition to the operand
    registers. Three formats are supported, selected by operand/scale dtype:

    * mxfp4  : e2m1 x e2m1, m16n8k64, ``kind::mxf4``,     scale_vec::2X, ue8m0
    * nvfp4  : e2m1 x e2m1, m16n8k64, ``kind::mxf4nvf4``, scale_vec::4X, ue4m3
    * mxfp8  : e4m3 x e4m3, m16n8k32, ``kind::mxf8f6f4``, scale_vec::1X, ue8m0
    """

    def __init__(self, m, n, k, a_dtype, b_dtype, cd_dtype, sf_dtype, scale_vec=None):
        self.scale_vec_int = scale_vec
        self.shape = (m, n, k)
        self.a_dtype = a_dtype if isinstance(a_dtype, str) else DTYPE_MAP[a_dtype]
        self.b_dtype = b_dtype if isinstance(b_dtype, str) else DTYPE_MAP[b_dtype]
        self.cd_dtype = cd_dtype if isinstance(cd_dtype, str) else DTYPE_MAP[cd_dtype]
        sf_dtype = sf_dtype if isinstance(sf_dtype, str) else DTYPE_MAP[sf_dtype]
        if sf_dtype not in SF_DTYPE_MAP:
            raise ValueError(f"Invalid scale-factor dtype for MXMMA: {sf_dtype}")
        self.sf_dtype = sf_dtype
        self.sf_ptx = SF_DTYPE_MAP[sf_dtype]

        if scale_vec is None:
            if self.a_dtype == "e2m1":
                scale_vec = 4 if self.sf_ptx == "ue4m3" else 2
            else:
                scale_vec = 1
        self.scale_vec_size = scale_vec

        if self.a_dtype == "e2m1":
            self.kind = "kind::mxf4nvf4"
            if scale_vec == 4:
                self.scale_vec = "scale_vec::4X"
            elif scale_vec == 2:
                self.scale_vec = "scale_vec::2X"
            else:
                raise ValueError(f"unsupported fp4 scale_vec: {scale_vec} (expected 2 or 4)")
        else:
            assert scale_vec == 1, "fp8/f6f4 microscale uses scale_vec::1X (group == mma K-tile)"
            self.kind = "kind::mxf8f6f4"
            self.scale_vec = "scale_vec::1X"

        self.native_mixed = self.kind == "kind::mxf8f6f4" and self.a_dtype != self.b_dtype
        b_reg_dtype = self.b_dtype
        if self.kind == "kind::mxf8f6f4" and DTYPE_BIT_WIDTH_MAP[self.b_dtype] < 8:
            b_reg_dtype = "e4m3"
        self.reg_a_count = calc_reg_count(m, k, self.a_dtype)
        self.reg_b_count = calc_reg_count(k, n, b_reg_dtype)
        self.reg_cd_count = calc_reg_count(m, n, self.cd_dtype)
        if self.cd_dtype == "f32":
            self.val_type_cd = "float"
            self.reg_cd_type = "float"
        elif self.cd_dtype == "f16":
            self.val_type_cd = "half"
            self.reg_cd_type = "uint32_t"
        elif self.cd_dtype == "bf16":
            self.val_type_cd = "nv_bfloat16"
            self.reg_cd_type = "uint32_t"
        else:
            raise ValueError(f"Invalid cd_dtype for MXMMA: {cd_dtype}")

    def to_cpp_str(self, include_class_name=False):
        reg_cd_type = self.reg_cd_type
        lines = [
            "static constexpr MmaType kMmaType = MmaType::MXMMA;",
            f"using MmaShape = Shape<{self.shape[0]}, {self.shape[1]}, {self.shape[2]}>;",
            "",
            f"using ValTypeC = {self.val_type_cd};",
            f"using ValTypeD = {self.val_type_cd};",
            "",
            f"static constexpr uint32_t kScaleVec = {self.scale_vec_int};",
            f"static constexpr uint32_t kATypeBits = {DTYPE_BIT_WIDTH_MAP[self.a_dtype]};",
            f"static constexpr uint32_t kBTypeBits = {DTYPE_BIT_WIDTH_MAP[self.b_dtype]};",
            f"static constexpr uint32_t kCTypeBits = {DTYPE_BIT_WIDTH_MAP[self.cd_dtype]};",
            f"static constexpr uint32_t kDTypeBits = {DTYPE_BIT_WIDTH_MAP[self.cd_dtype]};",
            f"static constexpr uint32_t kSFTypeBits = {DTYPE_BIT_WIDTH_MAP[self.sf_dtype]};",
            f"static constexpr bool kNativeMixed = {'true' if self.native_mixed else 'false'};",
            "",
            f"using ARegisters = uint32_t[{self.reg_a_count}];",
            f"using BRegisters = uint32_t[{self.reg_b_count}];",
            f"using CRegisters = {self.reg_cd_type}[{self.reg_cd_count}];",
            f"using DRegisters = {self.reg_cd_type}[{self.reg_cd_count}];",
            "",
            "CUDA_INLINE",
            f"static void fma(uint32_t *a, uint32_t *b, uint32_t sfa, uint32_t sfb, "
            f"{reg_cd_type} *c, {reg_cd_type} *d, "
            f"uint32_t byte_id_a, uint32_t thread_id_a, uint32_t byte_id_b, uint32_t thread_id_b) {{",  # noqa
            *self.generate_ptx(indent=2).strip("\n").split("\n"),
            "};",
        ]

        code = "\n".join("  " + x if x else x for x in lines)
        if include_class_name:
            code = f"class MmaOpClass {{\n{code}\n}};"

        return code

    def generate_ptx(self, indent=0):
        a_dtype = self.a_dtype
        b_dtype = self.b_dtype
        cd_dtype = self.cd_dtype
        shape = self.shape

        asm_op = f"mma.sync.aligned.{self.kind}.block_scale.{self.scale_vec}"
        asm_op += f".m{shape[0]}n{shape[1]}k{shape[2]}.row.col"
        asm_op += f".{cd_dtype}.{a_dtype}.{b_dtype}.{cd_dtype}.{self.sf_ptx}"

        counter = 0

        def take(count):
            nonlocal counter
            group = ", ".join(f"%{counter + i}" for i in range(count))
            counter += count
            return "{" + group + "}"

        d_group = take(self.reg_cd_count)
        a_group = take(self.reg_a_count)
        b_group = take(self.reg_b_count)
        c_group = take(self.reg_cd_count)
        sfa_group = take(1)
        sel_a_group = take(2)
        sfb_group = take(1)
        sel_b_group = take(2)

        t = "f" if cd_dtype == "f32" else "r"
        d_params = ", ".join(f'"+{t}"(d[{i}])' for i in range(self.reg_cd_count))
        a_params = ", ".join(f'"r"(a[{i}])' for i in range(self.reg_a_count))
        b_params = ", ".join(f'"r"(b[{i}])' for i in range(self.reg_b_count))
        c_params = ", ".join(f'"{t}"(c[{i}])' for i in range(self.reg_cd_count))

        asm_code = f"""
        asm volatile(
          "{asm_op} "
          "{d_group}, "
          "{a_group}, "
          "{b_group}, "
          "{c_group}, "
          "{sfa_group}, {sel_a_group}, "
          "{sfb_group}, {sel_b_group};\\n"
          : {d_params}
          : {a_params},
            {b_params},
            {c_params},
            "r"(sfa), "h"((uint16_t)byte_id_a), "h"((uint16_t)thread_id_a),
            "r"(sfb), "h"((uint16_t)byte_id_b), "h"((uint16_t)thread_id_b)
        );
        """

        space_count = len(re.findall("^\n( +)", asm_code)[0])
        asm_code = asm_code.replace("\n" + " " * space_count, "\n").strip()
        asm_code = "".join("\n" + " " * indent + x for x in asm_code.split("\n"))

        return asm_code


class MmaOpClass:
    @classmethod
    def from_config(cls, mma_type, m, n, k, a_dtype, b_dtype, cd_dtype, sf_dtype=None, scale_vec=None):
        mma_type = mma_type if isinstance(mma_type, MmaType) else getattr(MmaType, mma_type.upper())

        if mma_type == MmaType.MMA:
            return MmaOpClassImpl(m, n, k, a_dtype, b_dtype, cd_dtype)
        elif mma_type == MmaType.WGMMA:
            return WgmmaOpClassImpl(m, n, k, a_dtype, b_dtype, cd_dtype)
        elif mma_type == MmaType.MXMMA:
            if sf_dtype is None:
                raise ValueError("MXMMA requires sf_dtype (block scale-factor dtype)")
            return MxMmaOpClassImpl(m, n, k, a_dtype, b_dtype, cd_dtype, sf_dtype, scale_vec=scale_vec)
        else:
            raise ValueError(f"Invalid MMA Type: {mma_type}")
