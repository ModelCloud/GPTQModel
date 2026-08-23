# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Independent process controller for Qwen3 acceptance measurements."""

from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import os
import secrets
import socket
import stat
import struct
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

CONTROLLER_SCHEMA = "qvq-acceptance-controller-v4"
CONTROLLER_STAGES = ("quantization_producer", "fresh_process_reload", "acceptance_evaluation")
_ENV_PREFIX = "GPTQMODEL_QVQ_CONTROLLER_"
VERIFIER_PRIVATE_KEY_ENV = "GPTQMODEL_QVQ_VERIFIER_PRIVATE_KEY"
TRUST_CONFIG_ENV = "GPTQMODEL_QVQ_TRUST_CONFIG"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_F_ADD_SEALS = getattr(fcntl, "F_ADD_SEALS", 1033)
_F_SEAL_ALL_WRITES = sum(getattr(fcntl, name, fallback) for name, fallback in (
    ("F_SEAL_SEAL", 0x0001), ("F_SEAL_SHRINK", 0x0002), ("F_SEAL_GROW", 0x0004), ("F_SEAL_WRITE", 0x0008)
))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _physical_path_without_symlinks(path: Path) -> Path:
    if not path.is_absolute():
        raise RuntimeError(f"trusted path is not absolute: {path}")
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        try:
            identity = os.lstat(current)
        except OSError as error:
            raise RuntimeError(f"trusted path component is unavailable: {current}") from error
        if stat.S_ISLNK(identity.st_mode):
            raise RuntimeError(f"trusted path traverses a symlink: {current}")
    return path.resolve(strict=True)


def _read_nofollow(path: Path, *, exact_mode: int | None = None) -> tuple[bytes, os.stat_result]:
    path = _physical_path_without_symlinks(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RuntimeError(f"trusted file cannot be opened safely: {path}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"trusted file is not regular: {path}")
        if before.st_uid != os.geteuid():
            raise RuntimeError(f"trusted file has unsafe ownership: {path}")
        if exact_mode is not None and stat.S_IMODE(before.st_mode) != exact_mode:
            raise RuntimeError(f"trusted file mode must be exactly {exact_mode:04o}: {path}")
        chunks = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
        ):
            raise RuntimeError(f"trusted file changed while being read: {path}")
        return b"".join(chunks), before
    finally:
        os.close(descriptor)


def _trusted_content_identity(path: Path, *, exact_mode: int | None = None) -> tuple[Path, str, os.stat_result]:
    resolved = _physical_path_without_symlinks(path)
    content, identity = _read_nofollow(resolved, exact_mode=exact_mode)
    return resolved, hashlib.sha256(content).hexdigest(), identity


def load_trust_config() -> dict[str, Any]:
    configured = os.environ.get(TRUST_CONFIG_ENV)
    if not configured or not Path(configured).is_absolute():
        raise RuntimeError(f"operator trust configuration is absent: set {TRUST_CONFIG_ENV} to an absolute path")
    resolved_config = _physical_path_without_symlinks(Path(configured))
    if resolved_config.is_relative_to(_REPO_ROOT.resolve()):
        raise RuntimeError("operator trust configuration must reside outside the candidate repository")
    raw, _identity = _read_nofollow(resolved_config, exact_mode=0o600)
    try:
        trust = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("operator trust configuration is malformed") from error
    required = {"schema", "verifier_public_key", "verifier_public_key_sha256", "python_executable",
                "python_executable_sha256", "openssl_executable", "openssl_executable_sha256",
                "acceptance_policy", "acceptance_policy_sha256", "quant_config", "quant_config_sha256",
                "stage_scripts"}
    if not isinstance(trust, dict) or set(trust) != required or trust.get("schema") != "qvq-acceptance-trust-v1":
        raise RuntimeError("operator trust configuration has an invalid closed schema")
    for path_key, digest_key in (("verifier_public_key", "verifier_public_key_sha256"),
                                 ("python_executable", "python_executable_sha256"),
                                 ("openssl_executable", "openssl_executable_sha256"),
                                 ("acceptance_policy", "acceptance_policy_sha256"),
                                 ("quant_config", "quant_config_sha256")):
        path, digest, file_identity = _trusted_content_identity(Path(trust[path_key]))
        if path_key == "verifier_public_key" and path.is_relative_to(_REPO_ROOT.resolve()):
            raise RuntimeError("operator verifier public key must reside outside the candidate repository")
        if digest != trust[digest_key]:
            raise RuntimeError(f"operator-trusted {path_key} identity is absent or mismatched")
        if path_key.endswith("executable") and (
            not file_identity.st_mode & stat.S_IXUSR or file_identity.st_mode & 0o022
        ):
            raise RuntimeError(f"operator-trusted {path_key} has unsafe executable mode")
    scripts = trust["stage_scripts"]
    if not isinstance(scripts, dict) or set(scripts) != set(CONTROLLER_STAGES):
        raise RuntimeError("operator trust configuration lacks exact stage script identities")
    for stage, identity in scripts.items():
        if not isinstance(identity, dict) or set(identity) != {"path", "sha256"}:
            raise RuntimeError(f"operator-trusted {stage} script identity is absent or mismatched")
        path, digest, file_identity = _trusted_content_identity(Path(identity["path"]))
        if digest != identity["sha256"] or file_identity.st_mode & 0o022:
            raise RuntimeError(f"operator-trusted {stage} script identity is absent or mismatched")
    return trust


def trusted_python_executable() -> str:
    return str(Path(load_trust_config()["python_executable"]).resolve())


def _acceptance_policy() -> dict[str, Any]:
    trust = load_trust_config()
    raw, _identity = _read_nofollow(Path(trust["acceptance_policy"]))
    policy = json.loads(raw)
    required = {
        "schema", "dense_model", "revision", "device", "layer_scope", "calibration_rows", "yaqa_rows",
        "validation_rows", "maximum_bpw", "score_min", "final_kl_max_nats",
    }
    if not isinstance(policy, dict) or set(policy) != required or policy.get("schema") != "qwen3-8b-qvq-acceptance-policy-v1":
        raise RuntimeError("locked acceptance policy has an invalid closed schema")
    return policy


def _controller_dataset_bundle(values: Mapping[str, str]) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Read frozen producer inputs and prove JSONL/manifest identity and disjointness independently."""

    evidence: dict[str, Any] = {}
    identity_sets: dict[str, set[str]] = {}
    content_sets: dict[str, set[str]] = {}
    snapshots: dict[str, bytes] = {}
    for name in ("calibration", "yaqa", "validation"):
        source = Path(values[f"--{name}-dataset"])
        manifest_path = source.with_suffix(".manifest.json")
        try:
            source_raw, _source_identity = _read_nofollow(source)
            manifest_raw, _manifest_identity = _read_nofollow(manifest_path)
            lines = source_raw.decode("utf-8").splitlines()
            manifest = json.loads(manifest_raw)
        except (RuntimeError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"controller cannot read trusted {name} JSONL/manifest inputs") from error
        expected_split = "yaqa_tuning" if name == "yaqa" else name
        if (
            len(lines) != 512
            or manifest.get("schema_version") != 1
            or manifest.get("count") != 512
            or manifest.get("split") != expected_split
        ):
            raise RuntimeError(f"controller {name} input does not contain exactly 512 canonical rows")
        identities: set[str] = set()
        content_hashes: set[str] = set()
        samples = manifest.get("samples")
        if not isinstance(samples, list) or len(samples) != 512:
            raise RuntimeError(f"controller {name} manifest census is invalid")
        for ordinal, (line, sample) in enumerate(zip(lines, samples)):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"controller {name} JSONL row {ordinal} is invalid") from error
            identity = row.get("identity") if isinstance(row, dict) else None
            if not isinstance(identity, str) or not identity:
                raise RuntimeError(f"controller {name} JSONL row {ordinal} lacks identity")
            content_hash = hashlib.sha256(
                json.dumps(row.get("content"), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            if sample != {"ordinal": ordinal, "identity": identity, "content_sha256": content_hash}:
                raise RuntimeError(f"controller {name} manifest row {ordinal} does not match JSONL content")
            identities.add(identity)
            content_hashes.add(content_hash)
        if len(identities) != 512 or len(content_hashes) != 512:
            raise RuntimeError(f"controller {name} inputs contain duplicate identity or content")
        identity_sets[name] = identities
        content_sets[name] = content_hashes
        evidence[name] = {
            "source": str(source), "config": None, "split": "train", "row_start": 0, "rows": 512,
            "content_sha256": hashlib.sha256(source_raw).hexdigest(), "identity_manifest": str(manifest_path),
            "identity_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(), "manifest_verified": True,
        }
        snapshots[str(source)] = source_raw
        snapshots[str(manifest_path)] = manifest_raw
    for left_index, left in enumerate(identity_sets):
        for right in tuple(identity_sets)[left_index + 1:]:
            if identity_sets[left] & identity_sets[right] or content_sets[left] & content_sets[right]:
                raise RuntimeError(f"controller producer inputs are not disjoint: {left}/{right}")
    return evidence, snapshots


def controller_dataset_evidence(values: Mapping[str, str]) -> dict[str, Any]:
    return _controller_dataset_bundle(values)[0]


def _sealed_snapshot_descriptors(snapshots: Mapping[str, bytes]) -> tuple[dict[str, int], list[int]]:
    mapping: dict[str, int] = {}
    descriptors: list[int] = []
    for index, (path, payload) in enumerate(snapshots.items()):
        descriptor = os.memfd_create(f"qvq-input-{index}", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
        os.write(descriptor, payload)
        os.lseek(descriptor, 0, os.SEEK_SET)
        fcntl.fcntl(
            descriptor,
            _F_ADD_SEALS,
            _F_SEAL_ALL_WRITES,
        )
        mapping[path] = descriptor
        descriptors.append(descriptor)
    return mapping, descriptors


def _parse_linux_proc_stat(value: str) -> tuple[int, str, int, int]:
    opening = value.find("(")
    closing = value.rfind(")")
    if opening <= 0 or closing <= opening or closing + 2 >= len(value):
        raise RuntimeError("Linux proc stat has malformed comm boundaries")
    try:
        pid = int(value[:opening].strip())
        remainder = value[closing + 2:].split()
        ppid = int(remainder[1])  # field 4; remainder begins at field 3 (state)
        start_time_ticks = int(remainder[19])  # field 22
    except (IndexError, ValueError) as error:
        raise RuntimeError("Linux proc stat lacks canonical PID fields") from error
    return pid, value[opening + 1:closing], ppid, start_time_ticks


def _observe_linux_process(pid: int) -> dict[str, Any]:
    proc = Path("/proc") / str(pid)
    try:
        observed_pid, comm, ppid, start_time_ticks = _parse_linux_proc_stat(
            (proc / "stat").read_text(encoding="ascii")
        )
        executable_link = proc / "exe"
        executable = Path(os.readlink(executable_link)).resolve(strict=True)
        descriptor = os.open(executable_link, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
        try:
            executable_identity = os.fstat(descriptor)
            digest = hashlib.sha256()
            while chunk := os.read(descriptor, 1024 * 1024):
                digest.update(chunk)
        finally:
            os.close(descriptor)
        cmdline = (proc / "cmdline").read_bytes()
    except OSError as error:
        raise RuntimeError(f"controller cannot independently inspect child PID {pid}") from error
    if not cmdline.endswith(b"\0"):
        raise RuntimeError(f"controller observed malformed cmdline for child PID {pid}")
    return {
        "pid": observed_pid, "comm": comm, "ppid": ppid, "start_time_ticks": start_time_ticks,
        "executable": str(executable), "executable_sha256": digest.hexdigest(),
        "executable_device": executable_identity.st_dev, "executable_inode": executable_identity.st_ino,
        "cmdline_sha256": hashlib.sha256(cmdline).hexdigest(),
    }


def validate_required_command(stage: str, command: Sequence[str]) -> dict[str, str]:
    """Require the exact trusted interpreter/script and a closed, ordered stage argv."""

    command = list(command)
    trust = load_trust_config()
    policy = _acceptance_policy()
    script_identity = trust["stage_scripts"].get(stage)
    if not script_identity or len(command) < 3:
        raise ValueError(f"controller stage is invalid: {stage!r}")
    if command[0] != str(Path(trust["python_executable"]).resolve()):
        raise ValueError(f"controller {stage!r} argv[0] is not the operator-trusted interpreter")
    if Path(command[1]).resolve() != Path(script_identity["path"]):
        raise ValueError(f"controller {stage!r} command substituted the required executable")
    tail = command[2:]
    if stage == CONTROLLER_STAGES[0]:
        option_names = (
            "--model", "--output", "--quant-config", "--calibration-dataset", "--calibration-rows",
            "--yaqa-dataset", "--yaqa-rows", "--validation-dataset", "--validation-rows", "--device",
        )
        expected_tail_length = len(option_names) * 2 + 1
        if len(tail) != expected_tail_length or tail[-1] != "--verify-qwen3-acceptance-payload-parity":
            raise ValueError("producer command has unknown, duplicate, missing, or extra flags")
        values = {name: tail[index * 2 + 1] for index, name in enumerate(option_names)}
        if any(tail[index * 2] != name for index, name in enumerate(option_names)):
            raise ValueError("producer command options are not in exact canonical order")
        fixed = {
            "--model": policy["dense_model"],
            "--quant-config": str(Path(trust["quant_config"]).resolve()),
            "--calibration-rows": str(policy["calibration_rows"]),
            "--yaqa-rows": str(policy["yaqa_rows"]),
            "--validation-rows": str(policy["validation_rows"]),
            "--device": policy["device"],
        }
    else:
        subcommand = "payload-hashes" if stage == CONTROLLER_STAGES[1] else "evaluate"
        option_names = (
            ("--checkpoint", "--device", "--output")
            if stage == CONTROLLER_STAGES[1]
            else ("--dense-model", "--revision", "--checkpoint", "--manifest-dir", "--validation-jsonl",
                  "--held-out-diagnostics-jsonl", "--diverse-jsonl", "--device", "--maximum-bpw", "--score-min",
                  "--final-kl-max-nats", "--output")
        )
        if len(tail) != 1 + len(option_names) * 2 or tail[0] != subcommand:
            raise ValueError(f"controller {stage!r} command substituted the required subcommand")
        if any(tail[1 + index * 2] != name for index, name in enumerate(option_names)):
            raise ValueError(f"controller {stage!r} options are not exact and canonical")
        values = {name: tail[1 + index * 2 + 1] for index, name in enumerate(option_names)}
        fixed = ({
            "--dense-model": policy["dense_model"],
            "--revision": policy["revision"],
            "--device": policy["device"],
            "--maximum-bpw": str(policy["maximum_bpw"]),
            "--score-min": str(policy["score_min"]),
            "--final-kl-max-nats": str(policy["final_kl_max_nats"]),
        } if stage == CONTROLLER_STAGES[2] else {"--device": policy["device"]})
    for flag, expected in fixed.items():
        if values.get(flag) != expected:
            raise ValueError(f"controller {stage!r} command has noncanonical {flag}")
    path_options = {
        "--output", "--quant-config", "--calibration-dataset", "--yaqa-dataset", "--validation-dataset",
        "--checkpoint", "--manifest-dir", "--validation-jsonl", "--held-out-diagnostics-jsonl", "--diverse-jsonl",
    }
    for flag in path_options & values.keys():
        if values[flag] != str(Path(values[flag]).expanduser().resolve()):
            raise ValueError(f"controller {stage!r} {flag} is not an absolute normalized path")
    return values


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _hash_frame(digest: Any, payload: bytes) -> None:
    digest.update(struct.pack("<Q", len(payload)))
    digest.update(payload)


def _new_identity() -> str:
    return secrets.token_hex(32)


def _openssl(args: Sequence[str], *, input_bytes: bytes | None = None) -> subprocess.CompletedProcess:
    executable = str(Path(load_trust_config()["openssl_executable"]).resolve())
    try:
        return subprocess.run(
            [executable, *args],
            input=input_bytes,
            check=True,
            capture_output=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError("OpenSSL Ed25519 controller signing is unavailable or failed") from error


def _pinned_public_key() -> bytes:
    trust = load_trust_config()
    value, _identity = _read_nofollow(Path(trust["verifier_public_key"]))
    if hashlib.sha256(value).hexdigest() != trust["verifier_public_key_sha256"]:
        raise RuntimeError("operator verifier public key fingerprint mismatches trusted deployment input")
    if not value.startswith(b"-----BEGIN PUBLIC KEY-----"):
        raise RuntimeError("pinned Qwen3 acceptance verifier public key is malformed")
    return value


def _load_trusted_signing_key() -> tuple[bytes, bytes, Path, os.stat_result, str]:
    configured = os.environ.get(VERIFIER_PRIVATE_KEY_ENV)
    if not configured:
        raise RuntimeError(f"trusted verifier signing authority is absent: set {VERIFIER_PRIVATE_KEY_ENV}")
    private_path = Path(configured)
    if not private_path.is_absolute():
        raise RuntimeError("trusted verifier private key must be an existing absolute external path")
    resolved_private_path = _physical_path_without_symlinks(private_path)
    if resolved_private_path.is_relative_to(_REPO_ROOT.resolve()):
        raise RuntimeError("trusted verifier private key must not reside in the candidate repository")
    private_key, identity = _read_nofollow(private_path, exact_mode=0o600)
    private_digest = hashlib.sha256(private_key).hexdigest()
    with TemporaryDirectory(prefix="qvq-controller-key-check-") as temporary:
        private_copy = Path(temporary) / "stable-private.pem"
        derived_path = Path(temporary) / "derived-public.pem"
        private_copy.write_bytes(private_key)
        private_copy.chmod(0o600)
        _openssl(["pkey", "-in", str(private_copy), "-pubout", "-out", str(derived_path)])
        derived_public = derived_path.read_bytes()
    pinned_public = _pinned_public_key()
    if derived_public != pinned_public:
        raise RuntimeError("trusted verifier private key does not match the pinned public trust root")
    return private_key, pinned_public, private_path, identity, private_digest


def _sign(private_key: bytes, payload: bytes) -> str:
    with TemporaryDirectory(prefix="qvq-controller-sign-") as temporary:
        key_path = Path(temporary) / "private.pem"
        message_path = Path(temporary) / "message.bin"
        signature_path = Path(temporary) / "signature.bin"
        key_path.write_bytes(private_key)
        message_path.write_bytes(payload)
        _openssl(
            [
                "pkeyutl",
                "-sign",
                "-inkey",
                str(key_path),
                "-rawin",
                "-in",
                str(message_path),
                "-out",
                str(signature_path),
            ]
        )
        return base64.b64encode(signature_path.read_bytes()).decode("ascii")


def verify_controller_signature(transcript: Mapping[str, Any]) -> bool:
    unsigned = dict(transcript)
    signature = unsigned.pop("controller_signature_ed25519", None)
    if not isinstance(signature, str):
        return False
    try:
        public_key = _pinned_public_key()
    except RuntimeError:
        return False
    if unsigned.get("verifier_public_key_sha256") != hashlib.sha256(public_key).hexdigest():
        return False
    try:
        signature_bytes = base64.b64decode(signature, validate=True)
    except (ValueError, TypeError):
        return False
    with TemporaryDirectory(prefix="qvq-controller-verify-") as temporary:
        public_path = Path(temporary) / "public.pem"
        message_path = Path(temporary) / "message.bin"
        signature_path = Path(temporary) / "signature.bin"
        public_path.write_bytes(public_key)
        message_path.write_bytes(_canonical(unsigned))
        signature_path.write_bytes(signature_bytes)
        try:
            _openssl(
                [
                    "pkeyutl",
                    "-verify",
                    "-pubin",
                    "-inkey",
                    str(public_path),
                    "-rawin",
                    "-in",
                    str(message_path),
                    "-sigfile",
                    str(signature_path),
                ]
            )
        except RuntimeError:
            return False
    return True


def controller_authority_receipt(transcript: Mapping[str, Any]) -> dict[str, Any]:
    public_key = _pinned_public_key()
    fingerprint = hashlib.sha256(public_key).hexdigest()
    if transcript.get("verifier_public_key_sha256") != fingerprint:
        raise ValueError("controller transcript does not identify the pinned verifier trust root")
    return {
        "schema": "qvq-acceptance-controller-trust-root-v4",
        "controller_instance_id": transcript.get("controller_instance_id"),
        "run_nonce": transcript.get("run_nonce"),
        "verifier_public_key_sha256": fingerprint,
        "transcript_sha256": hashlib.sha256(_canonical(transcript)).hexdigest(),
    }


def controller_environment() -> dict[str, str]:
    keys = (
        "RUN_NONCE", "CONTROLLER_INSTANCE_ID", "CONTROLLER_PID", "STAGE", "STAGE_NONCE",
        "PROCESS_INSTANCE_ID", "EVENT_FD",
    )
    values = {key.lower(): os.environ.get(_ENV_PREFIX + key) for key in keys}
    if any(value is None for value in values.values()):
        raise RuntimeError("acceptance stage was not spawned by the independent controller")
    return {key: str(value) for key, value in values.items()}


def _read_exact(reader: Any, count: int) -> bytes:
    chunks = []
    remaining = count
    while remaining:
        chunk = reader.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _receive_live_payload(reader: Any) -> dict[str, Any]:
    from gptqmodel.utils.qvq_acceptance import (
        QVQ_PAYLOAD_HASH_SCHEME,
        canonical_packed_tensor_schema,
        expected_cells,
        expected_projection_name,
        qwen3_payload_aggregate,
    )

    expected_modules = [expected_projection_name(*cell) for cell in expected_cells()]
    module_hashes: dict[str, str] = {}
    tensor_counts: dict[str, int] = {}
    current_module = None
    current_digest = None
    previous_key = None
    seen_full_names: set[str] = set()
    required_full_names: set[str] = set()
    cells = expected_cells()
    for cell in cells:
        required_full_names.update(canonical_packed_tensor_schema(*cell))
    while True:
        length_raw = _read_exact(reader, 8)
        if len(length_raw) != 8:
            raise RuntimeError("live payload stream ended before its frame boundary")
        header_length = struct.unpack("<Q", length_raw)[0]
        if header_length == 0:
            break
        if header_length > 1024 * 1024:
            raise RuntimeError("live payload metadata frame is unreasonably large")
        header_raw = _read_exact(reader, header_length)
        if len(header_raw) != header_length:
            raise RuntimeError("live payload metadata frame is truncated")
        header = json.loads(header_raw)
        if not isinstance(header, dict) or set(header) != {"module", "tensor", "dtype", "shape", "byte_count"}:
            raise RuntimeError("live payload metadata schema is invalid")
        module_name = header["module"]
        tensor_name = header["tensor"]
        key = (str(module_name).encode("utf-8"), str(tensor_name).encode("utf-8"))
        if previous_key is not None and key <= previous_key:
            raise RuntimeError("live payload records are duplicated or not canonical")
        previous_key = key
        if module_name not in expected_modules or not isinstance(header["byte_count"], int) or header["byte_count"] < 0:
            raise RuntimeError("live payload module/byte census is invalid")
        dtype_bytes = {
            "torch.bool": 1, "torch.uint8": 1, "torch.int8": 1, "torch.int16": 2, "torch.int32": 4,
            "torch.int64": 8, "torch.float16": 2, "torch.bfloat16": 2, "torch.float32": 4, "torch.float64": 8,
        }
        shape = header["shape"]
        if (
            header["dtype"] not in dtype_bytes
            or not isinstance(shape, list)
            or any(not isinstance(size, int) or isinstance(size, bool) or size < 0 for size in shape)
        ):
            raise RuntimeError("live payload tensor dtype/shape is invalid")
        expected_bytes = dtype_bytes[header["dtype"]]
        for size in shape:
            expected_bytes *= size
        if header["byte_count"] != expected_bytes:
            raise RuntimeError("live payload tensor byte count disagrees with dtype/shape")
        raw = _read_exact(reader, header["byte_count"])
        if len(raw) != header["byte_count"]:
            raise RuntimeError("live payload tensor bytes are truncated")
        if module_name != current_module:
            if current_module is not None:
                module_hashes[current_module] = current_digest.hexdigest()
            current_module = module_name
            current_digest = hashlib.sha256()
            _hash_frame(current_digest, QVQ_PAYLOAD_HASH_SCHEME.encode("utf-8"))
            _hash_frame(current_digest, module_name.encode("utf-8"))
            tensor_counts[module_name] = 0
        required = canonical_packed_tensor_schema(*cells[expected_modules.index(module_name)])
        full_name = f"{module_name}.{tensor_name}"
        seen_full_names.add(full_name)
        if full_name in required:
            dtype_map = {"I32": "torch.int32", "F32": "torch.float32", "U8": "torch.uint8"}
            expected = required[full_name]
            if header["shape"] != expected["shape"] or header["dtype"] != dtype_map[expected["dtype"]]:
                raise RuntimeError(f"live payload required tensor metadata mismatch: {full_name}")
        _hash_frame(current_digest, tensor_name.encode("utf-8"))
        _hash_frame(current_digest, header["dtype"].encode("ascii"))
        _hash_frame(current_digest, json.dumps(header["shape"], separators=(",", ":")).encode("ascii"))
        _hash_frame(current_digest, raw)
        tensor_counts[module_name] += 1
    if current_module is not None:
        module_hashes[current_module] = current_digest.hexdigest()
    if list(module_hashes) != expected_modules:
        raise RuntimeError("live payload stream does not contain the canonical 252-module census")
    if not required_full_names <= seen_full_names:
        raise RuntimeError("live payload stream omits required canonical packed tensors")
    return {
        "scheme": QVQ_PAYLOAD_HASH_SCHEME,
        "module_count": len(module_hashes),
        "module_sha256": module_hashes,
        "module_tensor_counts": tensor_counts,
        "aggregate_sha256": qwen3_payload_aggregate(module_hashes, tensor_counts),
    }


def emit_controller_measurement(
    stage: str,
    measurement: Mapping[str, Any],
    *,
    live_payload_records: Any = None,
) -> dict[str, Any]:
    """Send one nonce-bound measurement to the parent and wait for its pre-return acknowledgement."""

    authority = controller_environment()
    if authority["stage"] != stage:
        raise RuntimeError(f"controller authorized {authority['stage']!r}, not {stage!r}")
    event = {
        "run_nonce": authority["run_nonce"],
        "controller_instance_id": authority["controller_instance_id"],
        "controller_pid": int(authority["controller_pid"]),
        "stage": stage,
        "stage_nonce": authority["stage_nonce"],
        "process_instance_id": authority["process_instance_id"],
        "measurement": dict(measurement),
    }
    descriptor = int(authority["event_fd"])
    with socket.socket(fileno=descriptor) as channel:
        channel.sendall(_canonical(event) + b"\n")
        if live_payload_records is not None:
            for header, raw in live_payload_records:
                encoded_header = _canonical(header)
                channel.sendall(struct.pack("<Q", len(encoded_header)))
                channel.sendall(encoded_header)
                channel.sendall(raw)
            channel.sendall(struct.pack("<Q", 0))
        acknowledgement = channel.makefile("rb").readline()
    try:
        acknowledged = json.loads(acknowledgement)
    except json.JSONDecodeError as error:
        raise RuntimeError("controller returned a malformed acknowledgement") from error
    if acknowledged.get("acknowledged_event_sha256") != _sha256(event) or (
        stage == CONTROLLER_STAGES[0]
        and acknowledged.get("validation_policy") != "qwen3-producer-pre-save-v1"
    ):
        raise RuntimeError("controller did not acknowledge the live stage measurement")
    return event


class AcceptanceController:
    """Own unpredictable identities and observe child measurements/spawn/exit facts."""

    def __init__(self) -> None:
        self.run_nonce = _new_identity()
        self.controller_instance_id = _new_identity()
        (
            self._private_key,
            public_key,
            self._private_key_path,
            self._private_key_identity,
            self._private_key_digest,
        ) = _load_trusted_signing_key()
        self._public_key_sha256 = hashlib.sha256(public_key).hexdigest()
        self.controller_pid = os.getpid()
        self.controller_parent_pid = os.getppid()
        self._records: list[dict[str, Any]] = []
        self._run_contract: dict[str, str] | None = None

    def spawn_stage(
        self,
        stage: str,
        command: Sequence[str],
        *,
        cwd: Path,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        if stage not in CONTROLLER_STAGES or any(record["stage"] == stage for record in self._records):
            raise ValueError(f"controller stage is invalid or replayed: {stage!r}")
        command_values = validate_required_command(stage, command)
        if stage == CONTROLLER_STAGES[0]:
            manifest_dir = str(Path(command_values["--calibration-dataset"]).parent)
            if (
                command_values["--yaqa-dataset"] != str(Path(manifest_dir) / "yaqa_tuning.jsonl")
                or command_values["--validation-dataset"] != str(Path(manifest_dir) / "validation.jsonl")
            ):
                raise ValueError("producer dataset paths do not share the canonical manifest directory")
            self._run_contract = {"checkpoint": command_values["--output"], "manifest_dir": manifest_dir}
        elif self._run_contract is None:
            raise ValueError("controller stage lacks the producer-owned run contract")
        elif command_values["--checkpoint"] != self._run_contract["checkpoint"]:
            raise ValueError("controller stage substituted the producer checkpoint path")
        if stage == CONTROLLER_STAGES[2]:
            manifest_dir = self._run_contract["manifest_dir"]
            expected_paths = {
                "--manifest-dir": manifest_dir,
                "--validation-jsonl": str(Path(manifest_dir) / "validation.jsonl"),
                "--held-out-diagnostics-jsonl": str(Path(manifest_dir) / "held_out_diagnostics.jsonl"),
                "--diverse-jsonl": str(Path(manifest_dir) / "diverse_32.jsonl"),
            }
            if any(command_values[key] != value for key, value in expected_paths.items()):
                raise ValueError("evaluation paths are split from the producer manifest/checkpoint contract")
        if stage == CONTROLLER_STAGES[0]:
            controller_datasets, dataset_snapshots = _controller_dataset_bundle(command_values)
            snapshot_mapping, snapshot_descriptors = _sealed_snapshot_descriptors(dataset_snapshots)
        else:
            controller_datasets, snapshot_mapping, snapshot_descriptors = None, {}, []
        stage_nonce = _new_identity()
        process_instance_id = _new_identity()
        parent_channel, child_channel = socket.socketpair()
        environment = os.environ.copy()
        environment.update(
            {
                _ENV_PREFIX + "RUN_NONCE": self.run_nonce,
                _ENV_PREFIX + "CONTROLLER_INSTANCE_ID": self.controller_instance_id,
                _ENV_PREFIX + "CONTROLLER_PID": str(self.controller_pid),
                _ENV_PREFIX + "STAGE": stage,
                _ENV_PREFIX + "STAGE_NONCE": stage_nonce,
                _ENV_PREFIX + "PROCESS_INSTANCE_ID": process_instance_id,
                _ENV_PREFIX + "EVENT_FD": str(child_channel.fileno()),
                _ENV_PREFIX + "DATASET_SNAPSHOTS": json.dumps({
                    "fds": snapshot_mapping,
                    "evidence": controller_datasets,
                }, sort_keys=True),
            }
        )
        validate_required_command(stage, command)
        spawned_ns = time.monotonic_ns()
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=environment,
            pass_fds=(child_channel.fileno(), *snapshot_descriptors),
        )
        child_channel.close()
        for descriptor in snapshot_descriptors:
            os.close(descriptor)
        parent_channel.settimeout(timeout)
        try:
            with parent_channel.makefile("rb") as reader:
                raw_event = reader.readline()
            if not raw_event:
                raise RuntimeError(f"controller stage {stage!r} exited without a measurement")
            event = json.loads(raw_event)
            expected = {
                "run_nonce": self.run_nonce,
                "controller_instance_id": self.controller_instance_id,
                "controller_pid": self.controller_pid,
                "stage": stage,
                "stage_nonce": stage_nonce,
                "process_instance_id": process_instance_id,
            }
            if not isinstance(event, dict) or any(event.get(key) != value for key, value in expected.items()):
                raise RuntimeError(f"controller stage {stage!r} returned unauthorized identity data")
            controller_live_payload = _receive_live_payload(reader) if stage == CONTROLLER_STAGES[0] else None
            validate_required_command(stage, command)
            os_process = _observe_linux_process(process.pid)
            trust = load_trust_config()
            expected_cmdline = b"\0".join(os.fsencode(item) for item in command) + b"\0"
            if (
                os_process["ppid"] != self.controller_pid
                or os_process["executable"] != str(Path(trust["python_executable"]).resolve())
                or os_process["executable_sha256"] != trust["python_executable_sha256"]
                or os_process["cmdline_sha256"] != hashlib.sha256(expected_cmdline).hexdigest()
            ):
                raise RuntimeError(f"controller stage {stage!r} OS process facts do not match authority")
            event_sha256 = _sha256(event)
            received_ns = time.monotonic_ns()
            acknowledgement = {"acknowledged_event_sha256": event_sha256}
            if stage == CONTROLLER_STAGES[0]:
                from gptqmodel.utils.qvq_acceptance import (
                    validate_producer_pre_save_measurement,
                )

                validate_producer_pre_save_measurement(
                    event["measurement"], event=event, controller_datasets=controller_datasets
                )
                if event["measurement"]["pre_save"]["payload"] != controller_live_payload:
                    raise RuntimeError("producer payload digests do not match controller-hashed live tensor bytes")
                acknowledgement["validation_policy"] = "qwen3-producer-pre-save-v1"
                acknowledgement["validated_before_save_monotonic_ns"] = time.monotonic_ns()
            parent_channel.sendall(_canonical(acknowledgement) + b"\n")
            acknowledged_ns = time.monotonic_ns()
        except BaseException:
            process.kill()
            process.wait()
            raise
        finally:
            parent_channel.close()
        exit_code = process.wait(timeout=timeout)
        exited_ns = time.monotonic_ns()
        record = {
            "stage": stage,
            "stage_nonce": stage_nonce,
            "process_instance_id": process_instance_id,
            "pid": process.pid,
            "parent_pid": os.getpid(),
            "os_process": os_process,
            "argv": list(command),
            "argv_sha256": hashlib.sha256(b"\0".join(os.fsencode(item) for item in command)).hexdigest(),
            "spawned_monotonic_ns": spawned_ns,
            "event_received_monotonic_ns": received_ns,
            "acknowledgement_sent_monotonic_ns": acknowledged_ns,
            "exited_monotonic_ns": exited_ns,
            "exit_code": exit_code,
            "event": event,
            "event_sha256": event_sha256,
            "acknowledgement": acknowledgement,
            "controller_datasets": controller_datasets,
            "controller_live_payload": controller_live_payload,
            "previous_record_sha256": self._records[-1]["record_sha256"] if self._records else None,
        }
        record["record_sha256"] = _sha256(record)
        self._records.append(record)
        if exit_code != 0:
            raise subprocess.CalledProcessError(exit_code, command)
        return record

    def signed_transcript(self, *, require_complete: bool = True) -> dict[str, Any]:
        try:
            current_bytes, current = _read_nofollow(self._private_key_path, exact_mode=0o600)
        except RuntimeError as error:
            raise RuntimeError("trusted verifier private-key path changed after controller initialization") from error
        expected = self._private_key_identity
        if (
            (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns, current.st_ctime_ns)
            != (expected.st_dev, expected.st_ino, expected.st_size, expected.st_mtime_ns, expected.st_ctime_ns)
            or hashlib.sha256(current_bytes).hexdigest() != self._private_key_digest
        ):
            raise RuntimeError("trusted verifier private-key path changed after controller initialization")
        actual_stages = tuple(record["stage"] for record in self._records)
        if (require_complete and actual_stages != CONTROLLER_STAGES) or actual_stages != CONTROLLER_STAGES[: len(actual_stages)]:
            raise RuntimeError("controller transcript stages are incomplete or out of order")
        transcript = {
            "schema": CONTROLLER_SCHEMA,
            "controller_instance_id": self.controller_instance_id,
            "run_nonce": self.run_nonce,
            "controller_pid": self.controller_pid,
            "controller_parent_pid": self.controller_parent_pid,
            "verifier_public_key_sha256": self._public_key_sha256,
            "trust_config_sha256": hashlib.sha256(_canonical(load_trust_config())).hexdigest(),
            "controller_datasets": (
                self._records[0]["controller_datasets"] if self._records else None
            ),
            "processes": list(self._records),
        }
        transcript["controller_signature_ed25519"] = _sign(self._private_key, _canonical(transcript))
        return transcript


__all__ = [
    "CONTROLLER_SCHEMA",
    "CONTROLLER_STAGES",
    "AcceptanceController",
    "controller_authority_receipt",
    "controller_dataset_evidence",
    "controller_environment",
    "emit_controller_measurement",
    "load_trust_config",
    "trusted_python_executable",
    "validate_required_command",
    "verify_controller_signature",
]
