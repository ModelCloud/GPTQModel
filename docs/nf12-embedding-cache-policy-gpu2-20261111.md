# NF12 embedding cache policy on physical GPU 2

## Decision

Keep `__stcs` for the SM80, K=5120, single-copy one-ID prefill CTA. It improves Lite's
M=3072 int64 code-ID lookup by 1.044–1.074× across independent same-weight
paired runs. Uniform random IDs are flat. Keep the 8-ID CTA's existing load
and store instructions: no tested policy gave it a reproducible gain. This
experiment did not touch the running Lite service.

The final change is confined to
[`embedding_kernel`](../gptqmodel_ext/nf12/csrc/codec/embedding.cu):
`Repeats == 1 && Grid2D && IdsPerCTA == 1` uses `__stcs` for both output vectors. The 8-ID
specialization retains default stores, the 16-ID specialization retains
`__stcg`, and decode and other architectures retain their original code. The
merged four-copy embedding specialization keeps its existing store policy.

## Method and scope

- Baseline: `e3141059`, untouched native source. Candidate and baseline C APIs
  were loaded in one process, shared the same packed pointers and ID tensors,
  and were called directly as `nf12_embedding_i64` or `nf12_embedding`.
- Hardware: physical GPU 2, NVIDIA PG506-230, SM80. Command environment:
  `CUDA_VISIBLE_DEVICES=2 PYTHON_GIL=0`; interpreter:
  `/root/venv-lite-torch215/bin/python`; PyTorch 2.15.0+cu132; nvcc 13.3.73.
- Weight: full released DeepSeek-V4.1-Flash-1017 `embed.weight`, BF16
  `[129280, 5120]`, SHA-256
  `aa1ac979edab508950a871c370fa645a875203a1a35c20ba9c1d45bb1ed46e3c`.
  Code IDs came from
  [`nf12-lite-cuda-token-ids-20261111.json`](nf12-lite-cuda-token-ids-20261111.json).
- Timing: 64 calls per CUDA graph, paired ABBA rounds with alternating order.
  A dense BF16 graph surrounded each pair. Uniform random graphs rotated 64
  independently sampled ID tensors. Screening used 12 rounds; confirmations
  and final runs used 32. Medians below are per call in microseconds. A ratio
  above 1 means the candidate was faster. The
  [raw report](nf12-embedding-cache-policy-gpu2-20261111.json) retains every
  timing sample, seed, weight hash, source hash, and memory observation.

The sweep changed only the three 128-bit packed source loads or only the two
128-bit output stores in the 8-ID and one-ID templates. It tried `.cg`, `.ca`,
and `.cs` loads, and `.cg`, `.wb`, and `.cs` stores. Escape loads, 16-ID CTAs,
geometry, dispatch, packing, and decode were unchanged.

## Screening result

Candidate / baseline throughput ratio for int64 IDs, in column order:

| Hint | M312 code | M312 random | M313 code | M313 random | M3072 code | M3072 random |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Packed load `.cg` | 0.849 | 1.010 | 0.851 | 1.013 | 0.853 | 1.006 |
| Packed load `.ca` | 0.997 | 0.999 | 0.994 | 0.999 | 1.001 | 1.003 |
| Packed load `.cs` | 0.991 | 1.010 | 0.987 | 1.010 | 0.947 | 0.970 |
| Output store `.cg` | 0.994 | 1.000 | 0.997 | 1.001 | 1.046 | 1.000 |
| Output store `.wb` | 0.998 | 1.001 | 0.997 | 1.002 | 1.004 | 1.003 |
| Output store `.cs` | 0.994 | 1.000 | 0.997 | 0.998 | 1.044 | 1.002 |

The identical-library control also measured 0.994× at M312 code IDs. The
final 8-ID SASS instruction sequence is byte-identical to baseline, so its
small observed offsets are measurement variation. `.cg` loads hurt code IDs
substantially. One-ID `.cg` stores improved code IDs by 1.046×, 1.022×, and
1.023× in three runs; `.cs` stores were more consistent.

## One-ID confirmation

These are separate processes using the same released weight and fixture. The
first three rows used a sweep variant that also changed 8-ID stores. The last
two used the final source, which changes only one-ID stores.

| Run | Rounds | M3072 int64 code: baseline → candidate | Speedup | M3072 random speedup |
| --- | ---: | ---: | ---: | ---: |
| Store `.cs`, seed 20261120 | 12 | 26.368 → 25.248 µs | 1.044× | 1.002× |
| Store `.cs`, seed 20261121 | 32 | 26.248 → 25.152 µs | 1.044× | 0.999× |
| Store `.cs`, seed 20261122 | 32 | 24.200 → 22.688 µs | 1.067× | 1.000× |
| Final source, full restore, seed 20261124 | 32 | 24.000 → 22.496 µs | 1.067× | 1.002× |
| Final source, physical memory, seed 20261126 | 32 | 24.048 → 22.400 µs | 1.074× | 1.000× |
| Final source, all rotating IDs verified, seed 20261127 | 32 | 23.984 → 22.544 µs | 1.064× | 1.000× |

The final int32 M3072 run also improved code IDs from 23.608 to 22.192 µs
(1.064×); its random IDs were 44.784 → 44.720 µs. On the final physical
memory run, M2049 int64 code IDs improved 15.600 → 15.200 µs (1.026×), while
M2049 and M4095 random IDs were within 0.2% of baseline. The M3072 dense BF16
lookup took 17.952 µs in that same run, so packed embedding still trails the
dense oracle for Lite prefill.

SM80 SASS confirms the intended instruction change. The one-ID template has
two `STG.E.EF.128` instructions where baseline had two `STG.E.128` instructions;
its three packed loads remain `LDG.E.128.CONSTANT`. The 8-ID template's
instruction sequence has the same hash as baseline
(`3b6cf49c18d67688`, first 16 SHA-256 hex digits of normalized SASS).

| Source hint | SM80 SASS for each targeted 128-bit operation |
| --- | --- |
| Baseline `__ldg` | `LDG.E.128.CONSTANT` |
| `__ldcg` / `__ldca` / `__ldcs` | `LDG.E.128.STRONG.GPU` / `LDG.E.128.STRONG.SM` / `LDG.E.EF.128` |
| Baseline store | `STG.E.128` |
| `__stcg` / `__stwb` / `__stcs` | `STG.E.128.STRONG.GPU` / `STG.E.128.STRONG.SM` / `STG.E.EF.128` |

## Exactness and storage

The full packed table restored every original BF16 bit in bounded 128-row
chunks. Direct native calls for int32 and int64 IDs matched independent
`torch.nn.functional.embedding` BF16 bits at M=312, 313, 2049, 3072, and
4095 where each ID distribution applied. Invalid IDs `-1`, `N`, and the dtype
maximum wrote exact +0.0 rows. The final int64 run separately checked all 64
random ID tensors captured per shape against dense BF16. Repeated module
forwards retained only
`nf12_packed` and `nf12_escapes`; the production Python API's selected
embedding test passed all 11 cases, including new M3072 int64 and M4095 int64
coverage.

Packed weight plus escapes occupied 995,314,416 bytes. The checkpoint BF16
weight occupied 1,323,827,200 bytes and was retained solely by this benchmark
as its independent oracle. The inference path added zero packed-cache bytes,
zero retained expanded-weight bytes, and zero dequantization scratch bytes.
`nvidia-smi` reported this benchmark process at a maximum of **3010 MiB on
physical GPU 2**, below the **3072 MiB** limit. The final benchmark's
post-init free-memory delta peaked at 2536 MiB. After it exited, GPU 2 returned
to the original 84,515 MiB used and 12,949 MiB free.
