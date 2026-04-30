# Komodo-CANN Fused Op Contract

This directory contains the repository-side contract for the optional
Komodo-CANN fused W4A16 Ascend C operator.

The Python runtime looks for one of these torch operators:

- `torch.ops.gptqmodel_komodo_cann.w4a16_matmul`
- `torch.ops.gptqmodel_komodo_cann.komodo_cann_w4a16_matmul`
- `torch.ops.npu.gptqmodel_komodo_cann_w4a16_matmul`
- `torch.ops.npu.komodo_cann_w4a16_matmul`

The operator must implement:

```text
(Tensor x,
 Tensor packed_weight,
 Tensor scales,
 Tensor offsets,
 Tensor? bias,
 int group_size,
 int split_k,
 int base_m,
 int base_n,
 int base_k) -> Tensor
```

The first target is GPTQ W4A16 FP16 output for `group_size` 0, 32, 64, and 128.
Group-16 remains on the native fallback path until it has a separate fused
grouped design.

`op_ir/komodo_cann_w4a16_matmul.json` can be passed to `msopgen` as a starting
point for an Ascend C project:

```bash
msopgen gen \
  -i gptqmodel_ext/komodo_cann/op_ir/komodo_cann_w4a16_matmul.json \
  -f pytorch \
  -c ai_core-ascend910b \
  -lan cpp \
  -out /tmp/komodo_cann_w4a16_op
```

After the generated project registers a torch-visible operator, use
`GPTQMODEL_KOMODO_CANN_FUSED_REQUIRE=1` while profiling so a missing op cannot
silently fall back to `npu_weight_quant_batchmatmul`.

## aclnnWeightQuantBatchMatmulV3 Probe

`wq_bmm_v3_probe.cpp` is a narrow C++ bridge that registers
`torch.ops.gptqmodel_komodo_cann.w4a16_matmul` and calls the raw
`aclnnWeightQuantBatchMatmulV3` two-stage API directly. It is a bring-up probe,
not the final fused Ascend C kernel: it proves the repo can access V3 with
Komodo's existing packed INT4 tensors.

Komodo-CANN can auto-load this bridge when requested:

```bash
GPTQMODEL_KOMODO_CANN_V3=1 \
GPTQMODEL_KOMODO_CANN_FUSED_REQUIRE=1 \
GPTQMODEL_KOMODO_CANN_FUSED_OP=gptqmodel_komodo_cann.w4a16_matmul \
python scripts/profile_komodo_cann_npu.py --mode cann --iters 3 --warmup 1
```

The managed extension can also be built explicitly:

```bash
python - <<'PY'
from gptqmodel import extension
extension.load("komodo_cann_v3")
PY
```

Run:

```bash
python scripts/probe_komodo_cann_v3.py \
  --device 0 \
  --rows 8 \
  --in-features 256 \
  --out-features 256
```

Validation on the local 910B host matched the native
`torch.ops.npu.npu_weight_quant_batchmatmul` path for group sizes `0`, `32`,
`64`, and `128`, including the optional bias path. A decode-shaped
`M=1,K=4096,N=4096,group_size=128` probe also matched exactly in the latest run.

`GPTQMODEL_KOMODO_CANN_INNER_PRECISE=auto|0|1` controls the CANN
`inner_precise` argument. The default `auto` rule is intentionally narrow:
`inner_precise=1` is selected only for q-like group-32 decode shapes
(`rows <= 16`, `K >= 4096`, and `K <= N <= 2K`). Current notes in
`hw/komodo_cann.md` record the measured liked and avoided shapes.

The V3 bridge also caches repeatable ACL executors and reusable workspace by
default:

```bash
GPTQMODEL_KOMODO_CANN_V3_EXECUTOR_CACHE=1    # default
GPTQMODEL_KOMODO_CANN_V3_WORKSPACE_CACHE=1   # default
```

Set either variable to `0` while isolating CANN runtime behavior or measuring
per-call setup overhead.
