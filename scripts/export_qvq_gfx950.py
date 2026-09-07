"""AOT-export QVQ's existing gfx950 inner kernel for non-Python consumers."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def export(
    output: Path, m: int, k: int, n: int, transition_bits: int, bank_alt_id: int
):
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    from gptqmodel.utils.qvq_amd import (
        _bank_mask,
        _launch_config,
        _qvq_p32_gemv_gfx950_kernel,
        _qvq_p32_gfx950_kernel,
        _use_gemv,
    )

    if min(m, k, n) <= 0 or k % 16 or n % 16 or transition_bits not in range(4, 9):
        raise ValueError(
            "positive M, K/N multiples of 16, transition bits 4..8 required"
        )
    if bank_alt_id not in ((0,) if transition_bits == 8 else (1, 2, 3)):
        raise ValueError("W4 requires bank 0; P32 requires alternate bank 1..3")
    constants = {
        "size_m": m,
        "size_k": k,
        "size_n": n,
        "transition_bits": transition_bits,
        "words_per_tile": 4 * transition_bits,
        "alternate_mask": 0
        if transition_bits == 8
        else _bank_mask(transition_bits, bank_alt_id),
    }
    if _use_gemv(m, n):
        fn = _qvq_p32_gemv_gfx950_kernel
        grid = m * triton.cdiv(n, 64)
        constants.update(
            block_n=64, num_pid_n=triton.cdiv(n, 64), xcd_swizzle=grid % 8 == 0
        )
        options = {"num_warps": 8, "num_stages": 1, "waves_per_eu": 0}
    else:
        fn = _qvq_p32_gfx950_kernel
        bm, bn, warps = _launch_config(m, n, k)
        bk = 64 if m >= 128 else (32 if m == 64 and n >= 10240 else 16)
        # Existing fused path iterates full K blocks. Smaller legal K uses K16.
        if k % bk:
            bk = 16
        grid = triton.cdiv(m, bm) * triton.cdiv(n, bn)
        constants.update(
            block_m=bm,
            block_n=bn,
            block_k=bk,
            num_pid_n=triton.cdiv(n, bn),
            xcd_swizzle=grid % 8 == 0,
        )
        options = {
            "num_warps": warps,
            "num_stages": 1 if bm == 1024 else (3 if m <= 64 else 2),
            "waves_per_eu": 0,
            "matrix_instr_nonkdim": 16,
            "kpack": 1,
        }
    signature = dict(zip(fn.arg_names[:5], ("*fp16", "*i32", "*fp16", "*u8", "*fp32")))
    signature.update({name: "constexpr" for name in constants})
    compiled = triton.compile(
        ASTSource(fn, signature, constexprs=constants),
        target=GPUTarget("hip", "gfx950", 64),
        options=options,
    )
    for name in ("global_scratch_size", "profile_scratch_size"):
        if getattr(compiled.metadata, name, 0):
            raise ValueError(f"ABI does not support {name}")
    output.mkdir(parents=True, exist_ok=False)
    binary = compiled.asm["hsaco"]
    (output / "kernel.hsaco").write_bytes(binary)
    for name in ("ttir", "ttgir", "llir", "amdgcn"):
        if name in compiled.asm:
            (output / f"kernel.{name}").write_text(compiled.asm[name])
    record = {
        "abi_version": 1,
        "operation_version": 1,
        "m": m,
        "k": k,
        "n": n,
        "transition_bits": transition_bits,
        "bank_alt_id": bank_alt_id,
        "grid_x": grid,
        "threads": options["num_warps"] * 64,
        "shared_bytes": compiled.metadata.shared,
        "symbol": compiled.metadata.name,
        "arch": "gfx950",
        "triton_version": triton.__version__,
        "sha256": hashlib.sha256(binary).hexdigest(),
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "kernel_source_sha256": hashlib.sha256(
            Path(fn.fn.__code__.co_filename).read_bytes()
        ).hexdigest(),
        "exporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "kernel_hash": compiled.hash,
        "options": options,
        "contract": "fp16 X, fp16 decoded W, fp32 accumulation/output; no scales or rotations",
    }
    (output / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    for name in ("m", "k", "n", "transition-bits", "bank-alt-id"):
        parser.add_argument(f"--{name}", type=int, required=True)
    args = parser.parse_args()
    export(args.output, args.m, args.k, args.n, args.transition_bits, args.bank_alt_id)
