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
    match = re.search(pattern, text, re.S)
    if not match:
        raise ValueError(f"cannot find {label}")
    return int(match.group(1))


def c_config_fields(header: str) -> list[str]:
    match = re.search(r"typedef struct \{(.*?)\}\s*QvqP32WgmmaRawConfig", header, re.S)
    if not match:
        raise ValueError("cannot find QVQ raw config struct")
    fields: list[str] = []
    for declaration in re.findall(r"uint32_t\s+([^;]+);", match.group(1)):
        fields.extend(part.strip().split()[-1] for part in declaration.split(","))
    return fields


def zig_config_fields(loader: str) -> list[str]:
    match = re.search(r"const P32WgmmaRawConfig = extern struct \{(.*?)\n\};", loader, re.S)
    if not match:
        raise ValueError("cannot find ZML raw config struct")
    return re.findall(r"^\s*([a-z_]+):\s*u32", match.group(1), re.M)


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

        pin_match = re.search(r'_QVQ_COMMIT\s*=\s*"([0-9a-f]{40})"', repo)
        if not pin_match:
            raise ValueError("cannot find immutable QVQ pin")
        pin = pin_match.group(1)
        head = git(qvq, "rev-parse", "HEAD").stdout.strip()
        pin_known = git(qvq, "cat-file", "-e", f"{pin}^{{commit}}").returncode == 0
        if pin_known:
            diff = git(qvq, "diff", "--quiet", pin, head, "--", *QVQ_PAYLOAD)
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
        check(
            "Bidirectional algorithm set",
            qvq_algorithms == xla_algorithms == policy_algorithms,
            f"QVQ={sorted(qvq_algorithms)}, XLA={sorted(xla_algorithms)}, policy={sorted(policy_algorithms)}",
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
