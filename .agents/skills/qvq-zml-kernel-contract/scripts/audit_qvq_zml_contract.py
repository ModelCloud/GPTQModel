#!/usr/bin/env python3
"""Audit the static QVQ raw-ABI/ZML integration contract."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

QVQ_PAYLOAD = (
    "gptqmodel_ext/qvq/BUILD.bazel",
    "gptqmodel_ext/qvq/qvq_hadamard_input_raw_abi.cu",
    "gptqmodel_ext/qvq/qvq_hadamard_input_raw_abi.h",
    "gptqmodel_ext/qvq/qvq_wgmma_cuda.cu",
    "gptqmodel_ext/qvq/qvq_wgmma_raw_abi.cu",
    "gptqmodel_ext/qvq/qvq_wgmma_raw_abi.h",
    "gptqmodel_ext/qvq/p32",
)


def read(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8")


def git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=root, text=True, capture_output=True, check=False
    )


def first_int(pattern: str, text: str, label: str) -> int:
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        raise ValueError(f"cannot find {label}")
    return int(match.group(1))


def bool_expression(text: str, name: str) -> str:
    match = re.search(rf"const bool {re.escape(name)}\s*=\s*(.*?);", text, re.DOTALL)
    return re.sub(r"\s+", " ", match.group(1)).strip() if match else ""


def c_config_fields(header: str, name: str = "QvqP32WgmmaRawConfig") -> list[str]:
    match = re.search(rf"typedef struct \{{([^{{}}]*)\}}\s*{re.escape(name)}", header, re.DOTALL)
    if not match:
        raise ValueError("cannot find QVQ raw config struct")
    fields: list[str] = []
    for declaration in re.findall(r"uint32_t\s+([^;]+);", match.group(1)):
        fields.extend(part.strip().split()[-1] for part in declaration.split(","))
    return fields


def zig_config_fields(loader: str) -> list[str]:
    match = re.search(r"const P32WgmmaRawConfig = extern struct \{(.*?)\n\};", loader, re.DOTALL)
    if not match:
        raise ValueError("cannot find ZML raw config struct")
    return re.findall(r"^\s*([a-z_]+):\s*u32", match.group(1), re.MULTILINE)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qvq-root", type=Path, required=True)
    parser.add_argument("--zml-root", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    qvq = args.qvq_root.resolve()
    zml = args.zml_root.resolve()
    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": passed, "detail": detail})

    try:
        header = read(qvq / "gptqmodel_ext/qvq/qvq_wgmma_raw_abi.h")
        raw = read(qvq / "gptqmodel_ext/qvq/qvq_wgmma_raw_abi.cu")
        input_header = read(qvq / "gptqmodel_ext/qvq/qvq_hadamard_input_raw_abi.h")
        input_raw = read(qvq / "gptqmodel_ext/qvq/qvq_hadamard_input_raw_abi.cu")
        qvq_build = read(qvq / "gptqmodel_ext/qvq/BUILD.bazel")
        repo = read(zml / "third_party/qvq/repo.bzl")
        loader = read(zml / "platforms/cuda/qvq/qvq.zig")
        policy = read(zml / "zml/qvq.zig")
        xla = read(zml / "third_party/xla/qvq-p32-xla-integration.patch")

        qvq_abi = first_int(r"QVQ_WGMMA_RAW_ABI_VERSION\s+(\d+)u?", header, "QVQ ABI")
        zig_abi = first_int(
            r"const P32WgmmaRawConfig = extern struct \{.*?abi_version: u32 = (\d+)",
            loader,
            "ZML ABI struct version",
        )
        runtime_abi = first_int(r"if \(version\(\) != (\d+)\)", loader, "ZML runtime ABI guard")
        check("ABI version", qvq_abi == zig_abi == runtime_abi, f"QVQ={qvq_abi}, Zig={zig_abi}, runtime={runtime_abi}")
        qvq_fields = c_config_fields(header)
        zig_fields = zig_config_fields(loader)
        check("ABI struct layout", qvq_fields == zig_fields, f"QVQ={qvq_fields}, Zig={zig_fields}")

        decode_abi = first_int(r"QVQ_P32_W3_DECODE_RAW_ABI_VERSION\s+(\d+)u?", header, "W3 decoder ABI")
        zig_decode_abi = first_int(
            r"pub const P32W3DecodeRawConfig = extern struct \{.*?abi_version: u32 = (\d+)",
            loader,
            "ZML W3 decoder ABI",
        )
        decode_fields = c_config_fields(header, "QvqP32W3DecodeRawConfig")
        zig_decode_struct = re.search(
            r"pub const P32W3DecodeRawConfig = extern struct \{(.*?)\n\};", loader, re.DOTALL
        )
        zig_decode_fields = (
            re.findall(r"^\s*([a-z_]+):\s*u32", zig_decode_struct.group(1), re.MULTILINE)
            if zig_decode_struct else []
        )
        decode_symbols = (
            "qvq_p32_w3_decode_raw_abi_version",
            "qvq_p32_w3_decode_raw_launch",
            "qvq_p32_w3_decode_raw_launch_plan",
        )
        check(
            "W3 transient decoder ABI and loader",
            decode_abi == zig_decode_abi == 1
            and decode_fields == zig_decode_fields
            and all(symbol in header and symbol in raw and symbol in loader for symbol in decode_symbols),
            f"versions={decode_abi}/{zig_decode_abi}, fields={decode_fields}/{zig_decode_fields}",
        )

        input_abi = first_int(
            r"QVQ_HADAMARD_INPUT_RAW_ABI_VERSION\s+(\d+)u?",
            input_header,
            "input Hadamard ABI",
        )
        input_struct = re.search(
            r"typedef struct \{(.*?)\}\s*QvqHadamardInputRawConfig",
            input_header,
            re.DOTALL,
        )
        zig_input_struct = re.search(
            r"pub const HadamardInputRawConfig = extern struct \{(.*?)\n\};",
            loader,
            re.DOTALL,
        )
        if not input_struct or not zig_input_struct:
            raise ValueError("cannot find input Hadamard raw ABI structs")
        input_fields = re.findall(r"uint32_t\s+([a-z_]+);", input_struct.group(1))
        zig_input_fields = re.findall(
            r"^\s*([a-z_]+):\s*u32", zig_input_struct.group(1), re.MULTILINE
        )
        zig_input_abi = first_int(
            r"pub const HadamardInputRawConfig = extern struct \{.*?abi_version: u32 = (\d+)",
            loader,
            "ZML input Hadamard ABI",
        )
        check(
            "Input Hadamard ABI and struct",
            input_abi == zig_input_abi == 2
            and input_fields == zig_input_fields
            and "raw_version() == 2" in loader,
            f"QVQ/ZML versions={input_abi}/{zig_input_abi}, fields={input_fields}/{zig_input_fields}",
        )

        pin_match = re.search(r'_QVQ_COMMIT\s*=\s*"([0-9a-f]{40})"', repo)
        if not pin_match:
            raise ValueError("cannot find immutable QVQ pin")
        pin = pin_match.group(1)
        head = git(qvq, "rev-parse", "HEAD").stdout.strip()
        pin_known = git(qvq, "cat-file", "-e", f"{pin}^{{commit}}").returncode == 0
        if pin_known:
            # Compare the pinned payload with the files actually being tested,
            # including uncommitted ABI edits in a development worktree.
            diff = git(qvq, "diff", "--quiet", pin, "--", *QVQ_PAYLOAD)
            payload_equal = diff.returncode == 0
            check(
                "Pinned QVQ payload",
                payload_equal,
                f"pin={pin}, QVQ_HEAD={head}, relevant_payload_equal={payload_equal}",
            )
        else:
            check("Pinned QVQ payload", False, f"pin {pin} is absent from the supplied QVQ clone; fetch origin")

        symbols = (
            "qvq_p32_wgmma_raw_abi_version",
            "qvq_p32_wgmma_raw_workspace_bytes",
            "qvq_p32_wgmma_raw_launch",
        )
        missing_symbols = [s for s in symbols if s not in header or s not in loader]
        check("Raw symbols exported and loaded", not missing_symbols, f"missing={missing_symbols}")
        input_symbols = (
            "qvq_hadamard_input_raw_abi_version",
            "qvq_hadamard_input_raw_launch",
        )
        missing_input_symbols = [
            symbol for symbol in input_symbols
            if symbol not in input_header or symbol not in input_raw or symbol not in loader
        ]
        check(
            "Input Hadamard exported and loaded",
            not missing_input_symbols,
            f"missing={missing_input_symbols}",
        )
        check(
            "Input Hadamard built and sparse-reachable",
            "qvq_hadamard_input_raw_object" in qvq_build
            and all(
                name in repo
                for name in (
                    "qvq_hadamard_input_raw_abi.cu",
                    "qvq_hadamard_input_raw_abi.h",
                )
            ),
            "requires object in shared library and both files in sparse checkout",
        )
        check(
            "Input Hadamard admission and custom call",
            all(
                term in policy
                for term in (
                    "input.dim(axis_) == 8192",
                    "rows == 960",
                    "qvq_cuda.hopperHadamardInputAvailable()",
                    "su.convert(.f16)",
                    "P32HadamardInputCall.register(platform)",
                    "zml_qvq_p32_hadamard_input_raw_cuda",
                )
            )
            and policy.split("fn p32HadamardInputRawFfiCall", 1)[1]
            .split("const P32HadamardInputCall", 1)[0]
            .count("output.prepared.ptr") == 2
            and "workspace may alias output" in input_header,
            "SM90 M960/N8192 FP16 call is reachable with output/workspace alias",
        )
        check(
            "Input Hadamard runtime counter",
            "p32_input_hadamard_ffi_call_count.fetchAdd(1" in policy
            and "input_hadamard: usize" in policy,
            "optimized FFI calls must be visible in runner telemetry",
        )

        sparse_paths = (
            "qvq_wgmma_cuda.cu",
            "qvq_wgmma_raw_abi.cu",
            "qvq_wgmma_raw_abi.h",
            "p32/**",
        )
        missing_sparse = [p for p in sparse_paths if p not in repo]
        check("Sparse checkout reaches kernel payload", not missing_sparse, f"missing={missing_sparse}")

        families = {
            1: {
                "name": "ordered_m16",
                "qvq": ("algorithm == 1", "block_m == 0", "block_n == 0", "m <= 16"),
                "zml": ("hopper_algorithm == 1", "hopper_block_m == 0", "hopper_block_n == 0", "m <= 16"),
                "policy": ("config.hopper_algorithm = 1", "if (m > 16) return config"),
            },
            2: {
                "name": "direct_m64",
                "qvq": ("algorithm == 2", "block_m == 64", "block_n == 64", "m == 64", "split_count == 1"),
                "zml": ("hopper_algorithm == 2", "hopper_block_m == 64", "hopper_block_n == 64", "m == 64", "split_count == 1"),
                "policy": ("config.hopper_algorithm = if (m == 64) 2 else 3", "config.hopper_block_n = 64"),
            },
            3: {
                "name": "direct_m128",
                "qvq": ("algorithm == 3", "block_m == 128", "block_n == 64", "m == 128", "split_count == 1"),
                "zml": ("hopper_algorithm == 3", "hopper_block_m == 128", "hopper_block_n == 64", "m == 128", "split_count == 1"),
                "policy": ("config.hopper_algorithm = if (m == 64) 2 else 3", "config.hopper_block_n = 64"),
            },
        }
        for algorithm, family in families.items():
            qvq_missing = [token for token in family["qvq"] if token not in raw]
            zml_missing = [token for token in family["zml"] if token not in xla]
            policy_missing = [token for token in family["policy"] if token not in policy]
            check(
                f"Algorithm {algorithm} {family['name']}",
                not (qvq_missing or zml_missing or policy_missing),
                f"QVQ_missing={qvq_missing}, XLA_missing={zml_missing}, policy_missing={policy_missing}",
            )

        qvq_algorithms = {int(value) for value in re.findall(r"c->algorithm\s*==\s*(\d+)", raw)}
        xla_algorithms = {int(value) for value in re.findall(r"hopper_algorithm\s*==\s*(\d+)", xla)} - {0}
        policy_algorithms = {
            int(value)
            for value in re.findall(
                r"(?:config\.)?hopper_algorithm\s*(?:==|=)\s*(\d+)", policy
            )
        } - {0}
        # Algorithm 4 is the grouped gate/up direct-FFI route, not a
        # single-projection XLA composite. Check that route explicitly while
        # requiring exact agreement for the singleton algorithms.
        grouped_ffi = (
            "config.algorithm == 4 and config.group_count == 2" in loader
            and "group_count == 2 and m == 128 and k == 2048 and n == 16384" in policy
            and "config.hopper_algorithm = 4" in policy
            and "const bool grouped_gate_up = c->algorithm == 4" in raw
        )
        check("Grouped algorithm 4 direct FFI", grouped_ffi, f"route_present={grouped_ffi}")
        check(
            "Bidirectional singleton algorithm set",
            qvq_algorithms - {4} == xla_algorithms == policy_algorithms - {4}
            and (4 in qvq_algorithms) == (4 in policy_algorithms) == grouped_ffi,
            f"QVQ={sorted(qvq_algorithms)}, XLA={sorted(xla_algorithms)}, policy={sorted(policy_algorithms)}",
        )
        qvq_bn128 = bool_expression(raw, "bn128")
        xla_bn128 = bool_expression(xla, "direct_m960_two_consumers")
        gate_bn128_guard = "c->block_m == 160 && c->k == 2048 && c->n == 8192"
        xla_gate_bn128_guard = (
            "config.hopper_block_m == 160 && k == 2048 && n == 8192"
        )
        gate_bn128_supported = gate_bn128_guard in qvq_bn128
        gate_bn128_policy = (
            "config.hopper_block_n = if (k == 2048 and n == 8192 "
            "and transition_bits == 6) 128 else 64"
        )
        m960_terms = {
            "QVQ BM80": "c->block_m == 80" in raw and "c->block_n == 64" in raw,
            "QVQ BM160": "c->block_m == 160" in raw
            and "c->transition_bits == 6" in raw,
            "QVQ BN128": "c->block_n == 128" in qvq_bn128
            and "c->block_m == 64" in qvq_bn128,
            "XLA reused rows": "direct_m960_reused_rows" in xla
            and "config.hopper_block_m == 80" in xla
            and "config.hopper_block_m == 160" in xla,
            "XLA BN128": "config.hopper_block_m == 64" in xla_bn128
            and (xla_gate_bn128_guard in xla_bn128) == gate_bn128_supported,
            "ZML reused-row policy": (
                "config.hopper_block_m = if (n == 512) 64 else if "
                "(k == 2048 and n == 8192 and transition_bits == 6) 160 else 80"
            ) in policy,
            "ZML W3 gate BN policy": (gate_bn128_policy in policy)
            == gate_bn128_supported,
            "ZML automatic policy": "ZML_QVQ_M960_WGMMA" not in policy,
        }
        check(
            "M960 specialized geometry coverage",
            all(m960_terms.values()),
            str(m960_terms),
        )
        # Check the guarded admission expression, not just a BM160 mention in
        # the patch: a bare BM160 token previously masked dense XLA fallback.
        qvq_r10 = bool_expression(raw, "ten_rows")
        xla_reused = bool_expression(xla, "direct_m960_reused_rows")
        qvq_r10_guard = (
            "c->block_m == 160 && c->transition_bits == 6 && "
            "c->k == 2048 && c->n == 8192"
        )
        xla_r10_guard = (
            "config.hopper_block_m == 160 && "
            "config.transition_bits == 6 && k == 2048 && n == 8192"
        )
        check(
            "M960 W3 gate R10 exact XLA admission",
            qvq_r10 == qvq_r10_guard and xla_r10_guard in xla_reused
            and "direct_m960_shape &&" in xla
            and "config.hopper_algorithm == 5" in xla,
            f"QVQ_R10={qvq_r10!r}, XLA_reused_rows={xla_reused!r}",
        )
        check(
            "M960 W3 gate BN128 rollout is paired",
            not gate_bn128_supported or (
                xla_gate_bn128_guard in xla_bn128
                and gate_bn128_policy in policy
                and "AdmitsProductionM960W3GateR10Bn128" in xla
                and "RejectsM960W3GateR5Bn128NearMiss" in xla
            ),
            f"QVQ_BN128={qvq_bn128!r}, XLA_BN128={xla_bn128!r}",
        )

        attrs = ("hopper_algorithm", "hopper_block_m", "hopper_block_n", "split_count")
        missing_attrs = [a for a in attrs if a not in policy or a not in xla]
        check("StableHLO/XLA geometry attributes", not missing_attrs, f"missing={missing_attrs}")
        check(
            "Hopper grouped fallback",
            "if (group_count != 1) return config" in policy,
            "grouped raw-WGMMA admission must remain explicit",
        )
        check(
            "Runtime dispatch telemetry",
            "hopper_core_run_count" in loader and "hopperCoreRunCount" in loader,
            "optimized execution must be observable",
        )
    except (FileNotFoundError, ValueError) as error:
        check("Audit setup", False, str(error))

    passed = all(bool(item["passed"]) for item in checks)
    report = {"passed": passed, "checks": checks}
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for item in checks:
            status = "PASS" if item["passed"] else "FAIL"
            print(f"{status:4} {item['name']}: {item['detail']}")
        print(f"RESULT: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
