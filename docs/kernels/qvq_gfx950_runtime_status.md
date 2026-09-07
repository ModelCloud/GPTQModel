# gfx950 framework-neutral runtime: work in progress

Kernel ownership is QVQ; ZML-Ultra only consumes the exported ABI and manages
compiler integration, caller-owned buffers, streams and graph lifetimes.

The experimental `qvq_gfx950_abi.h` / `qvq_gfx950_runtime.cc` runtime loads
QVQ-produced Triton HSACO during explicit preparation. Execution launches the
prepared function without Python, allocation or module loading. The exporter
reuses the existing QVQ gfx950 kernels rather than duplicating their math.

## Executed evidence (2026-09-07)

- QVQ base: `66565c27`; runtime/export changes are uncommitted WIP.
- Device: MI355X VF, gfx950, PCI `0000:83:00.0`.
- Artifact: `/home/ubuntu/qvq-gfx950-runtime/m1-k256-n256-w3/manifest.json`.
- Shape: M=1, K=256, N=256, transition bits=6, alternate bank=3.
- Input FP16, output FP32, deterministic synthetic packed words and bank bytes.
- Public HIP ABI eager output exactly matched existing QVQ uncached execution.
- Three HIP graph replays, changing input contents at stable addresses between
  replays, each exactly matched that same implementation (max drift 0).
- Test process exited 0. Torch generated fixtures and reference results; the
  tested runtime launched its prepared module through the C ABI.

This is narrow ABI parity evidence, not an independent canonical accuracy test,
model-quality result, ZML/PJRT graph-safety proof or performance certification.
Invalid descriptors and lifecycle failures still require dedicated tests. Artifact
validation and provenance need hardening before untrusted artifacts are accepted.
No full GSM8K-Platinum score is available yet.

## Expanded checks

`tests/test_qvq_gfx950_abi.py` now exercises a non-default HIP stream,
eager execution and three changed-input replays, verifies the HSACO SHA256,
and checks independent planar reconstruction followed by FP32 matmul under
the thread's mean <= 0.003 / max <= 0.006 limits. Executed successfully:

- M=1,K=256,N=256, canonical W4 (transition bits=8, bank=0).
- M=1,K=5120,N=17408, P32 W3 (transition bits=6, alternate bank=3).

Both pytest processes exited 0, one test passed each. The latter is a real
Qwen projection geometry but still uses synthetic tensors, not model weights
or held-out activations. These results do not establish model quality.

The geometry suite subsequently passed all 18 synthetic cases: nine checkpoint
geometry/rate/bank combinations at M=1 and M=8, including canonical W4. The
full checkpoint's M=1/M=2048 artifact set contains 18 specializations, with all
400 projection metadata/bank bindings validated by the consuming ZML loader.
M=2048 independent kernel accuracy remains open.

ZML's public PJRT eager smoke passed changed inputs with exact expected outputs
16, 32, and 48. A native full-model smoke observed 400 ABI callbacks for each
prefill/decode pass. Evalution's first four GSM8K-Platinum CoT test rows scored
4/4 through native ZML (no Torch model inference); this is a small integration
diagnostic, not a full quality score. The full 1209-row evaluation is in progress.
PJRT command-buffer compatibility remains disabled pending actual replay proof.

## Build the host runtime

From this repository with a compatible ROCm installation:

```sh
g++ -std=c++17 -O2 -fPIC -shared -D__HIP_PLATFORM_AMD__ \
  -I/opt/rocm/include gptqmodel_ext/qvq/p32/qvq_gfx950_runtime.cc \
  -L/opt/rocm/lib -Wl,-rpath,/opt/rocm/lib -lamdhip64 \
  -o /absolute/output/path/libqvq_gfx950.so
```

The library and HSACO artifacts are trusted executable code. SHA256 checking
detects a mismatch with a trusted manifest; it does not authenticate an arbitrary
manifest or make untrusted HSACO safe. Preparation requires the exact exported
descriptor and symbol. The caller retains buffers and plans until asynchronous
work and all referencing graphs finish. No production default is changed.
