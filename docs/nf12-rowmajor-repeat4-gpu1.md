# Packed row-major NF12 fused repeat4 on GPU1 and GPU3

This page records the repeat4 prototype. Production embedding lookup uses the
[row-major embedding storage](nf12-row-major-embedding.md) flow; repeat4
remains a comparison path.

This final bounded prototype combines the [row-major packed
layout](nf12-rowmajor-layout-prototype-gpu1.md) with a native kernel that
decodes each selected 64-value row block once and writes four exact BF16
copies. Production dispatch and Lite are unchanged. The comparator is the
actual tuned `gptqmodel_nf12.embedding_repeat4` public op at
`/root/nf12-fused-embed-gpu7`, commit
`1009d03f84401d8838fdd478052aeec0043fea08`.

## Validation and measurement

- Physical GPU1, NVIDIA PG506-232 SM80; `CUDA_VISIBLE_DEVICES=1`,
  `PYTHON_GIL=0`, `/root/venv-lite-torch215/bin/python`.
- Full released `embed.weight` `[129280,5120]`, SHA256
  `aa1ac979edab508950a871c370fa645a875203a1a35c20ba9c1d45bb1ed46e3c`.
  Code IDs are the Lite fixture prefix; random int64 IDs use GPU seed
  `20261111 + M`. Both kernels used the same canonical source and IDs in each
  pair. The row-major table was explicitly transformed from canonical NF12.
- Every full-size code and random output matched an independent CPU-sourced
  dense BF16 lookup repeated four times, bit for bit. Both kernels returned
  four exact `+0.0` rows for negative and out-of-range int64 IDs, including
  magnitudes of `2**40`.
- Targeted tests use a forced `[16,128]` fixture with 254 selected-half escape
  collisions, original escapes, signed zero, subnormals, and a NaN payload.
  Every repeat4 store policy returns exact BF16 bits for int32 and int64 IDs.
  After first and repeated forwards, the row-major object still owns exactly
  one uint8 packed tensor and one int16 escape tensor; no dynamic layout or
  decoded-weight cache is installed.
- Eight-call CUDA graphs and paired ABBA ordering. Both confirmation runs have
  16 rounds and 32 samples per side. All 16 long-M code-ID paired differences
  favor row-major on both GPUs. For random IDs at M≥1536, GPU3 has 16/16
  favorable rounds in every case and GPU1 has at least 15/16. The
  [GPU1 raw confirmation](nf12-rowmajor-repeat4-confirm-gpu1-20261111.json)
  and independent [GPU3 raw report](nf12-rowmajor-repeat4-gpu3-20261111.json)
  contain every sample, paired difference, exactness check, and source hash.

The winning long-row variant uses a four-lane register transpose before its
stores. Each store instruction then writes adjacent 16-byte chunks across the
four lanes. The direct four-lane stores were 3–29 µs slower than tuned HC4 in
the [initial paired run](nf12-rowmajor-repeat4-gpu1-20261111.json); the
coalesced-store [comparison run](nf12-rowmajor-repeat4-coalesced-gpu1-20261111.json)
reversed that result. This is a store-layout effect, not an extra weight
decode or a persistent expanded representation.

## Confirmed paired medians (microseconds)

Each pair is tuned canonical HC4 / row-major fused HC4. The GPU3 run was
performed independently by the user with the reproduction command below.
Both reports identify the same released weight, tuned upstream source commit,
and row-major kernel hash.

| IDs | M | GPU1 tuned / row | GPU3 tuned / row | GPU1 ratio | GPU3 ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Code | 312 | 9.728 / 9.984 | 8.832 / 9.088 | 0.974× | 0.972× |
| Code | 313 | 9.728 / 10.112 | 8.704 / 9.088 | 0.962× | 0.958× |
| Code | 1536 | 39.808 / 34.496 | 38.976 / 32.000 | 1.154× | 1.218× |
| Code | 2048 | 53.248 / 44.608 | 51.584 / 41.408 | 1.194× | 1.246× |
| Code | 3072 | 82.432 / 71.808 | 80.128 / 70.272 | 1.148× | 1.140× |
| Random | 312 | 9.728 / 10.112 | 8.832 / 9.216 | 0.962× | 0.958× |
| Random | 313 | 9.792 / 10.112 | 8.960 / 9.216 | 0.968× | 0.972× |
| Random | 1536 | 50.560 / 40.832 | 53.888 / 42.240 | 1.238× | 1.276× |
| Random | 2048 | 66.112 / 57.216 | 67.392 / 57.024 | 1.155× | 1.182× |
| Random | 3072 | 97.920 / 88.320 | 98.816 / 88.448 | 1.109× | 1.117× |

An earlier GPU1 16-round pass with the same kernel measured code M1536
`38.656 / 31.808` µs and M3072 `79.616 / 70.208` µs; random M1536 was
`50.560 / 40.704` µs and random M3072 `98.048 / 88.384` µs. Absolute
latency changed between runs, while the long-M paired direction persisted.

Geometry: M312/313 uses 128 threads, 8 IDs/CTA, coalesced `.cg` stores;
M1536/2048 uses 128 threads, 1 ID/CTA, coalesced `.cs` stores; M3072 uses
64 threads, 1 ID/CTA, coalesced `.cg` stores. The earlier bounded pass also
tested the tuned canonical geometry's direct-store analogue and one prior
row-major vector geometry. It did not repeat agent7's broader fused CTA sweep.

On this released table, canonical and row-major each retain **995,314,416
packed bytes**; selected-half collisions are zero. Benchmarking kept both
packed layouts resident for pairing, with no persistent expanded weight.
Peak PyTorch allocated/reserved GPU memory on both confirmation runs was
2,319,788,544 / 2,420,113,408 bytes; the process allocator was capped at
2.5% of the selected GPU, and each run stayed below 3 GiB additional VRAM.
The HC4 output itself is
`M × 4 × 5120 × 2` bytes, reaching 125,829,120 bytes at M3072.

## Decision and lifecycle

For this embedding-only released table, **the long-prefill gain is large
enough to justify an integration study**: 1.14–1.25× on code IDs at
M1536/2048/3072 across GPU1 and GPU3, with the same packed resident size if
row-major replaces canonical NF12. Short M312/313 is about 3–5% slower. The one-time GPU row transform
measured about 25 ms in the preceding load benchmark. A simple division by
the measured code-ID savings suggests roughly 2,400–4,700 long-prefill
lookups to amortize that conversion when starting from canonical packed NF12;
a preconverted row-major checkpoint would avoid it.

This does not establish a full Lite service win. The released checkpoint has
separate `embed.weight` and `head.weight`, so an embedding-only replacement is
possible, but a generic tied linear head still needs canonical NF12 or another
qualified layout. Keeping both layouts for one weight adds 995,314,416 GPU
bytes. The row-major helper is an explicit copy with versioned checkpoint
keys; it has **no dynamic layout cache**, mutation hook, or production dispatch.
No Lite service or PR was changed for this experiment.

## Independent GPU3 reproduction

The user completed this independent run on physical GPU3. The exact command
for the selected coalesced variant is:

```bash
cd /root/nf12-next-gpu1
CUDA_VISIBLE_DEVICES=3 PYTHON_GIL=0 PYTHONPATH=/root/nf12-next-gpu1 \
  /root/venv-lite-torch215/bin/python scripts/bench_nf12_rowmajor_repeat4_sm80.py \
  --physical-gpu 3 --variants coalesced_store --repeat 8 --rounds 16 \
  --json-out docs/nf12-rowmajor-repeat4-gpu3-20261111.json
```

The command loads the tuned public HC4 operator from the GPU7 checkout in the
same process and writes its report in this worktree. Its SM80 and one-visible-
GPU guards reject a wrong assignment.
On another host, set `NF12_TUNED_CHECKOUT` to a checkout of GPTQModel
PR #3777 before running the command.

Source: [experimental API](../gptqmodel_nf12/experimental_rowmajor.py),
[native kernel](../gptqmodel_ext/nf12/csrc/torch/rowmajor_embedding.cu),
[benchmark](../scripts/bench_nf12_rowmajor_repeat4_sm80.py), and
[tests](../tests/test_nf12_rowmajor_experimental.py).
