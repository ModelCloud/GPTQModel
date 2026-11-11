# NF12 row-major embedding layout prototype on GPU1

Baseline: `e3141059`. This is an explicit experimental helper and native
prototype; the production NF12 operator, Lite loader, and Lite model are
unchanged. It extends the [four-lane lookup experiment](nf12-rowmajor-embedding-next-gpu1.md)
with CPU and GPU transforms from canonical packed NF12.

## Layout and API

`gptqmodel_nf12.experimental_rowmajor.from_nf12` accepts a canonical
`NF12Weight` on CPU or NVIDIA SM80 and returns `NF12RowMajorWeight`. The object
holds only uint8 row-major packed bytes, four-BF16-value int16 escapes, and
scalar metadata. `experimental_rowmajor.embedding` performs exact int32/int64
lookup with explicit four-lane CTA geometry. The layout has separate,
versioned `nf12_row_*` state-dict keys; a small safetensors save/load roundtrip
is tested. It cannot be passed to the canonical NF12 linear or embedding ops.

The CPU transform uses bounded NumPy chunks. The GPU transform counts
selected-half sentinel collisions per canonical super-atom, prefix-sums one
count per tile, and packs rows with stable collision indices. Its temporary
GPU count, prefix, and error tensors total 7,756,804 bytes for this weight;
no BF16 weight tile or expanded code tensor is retained. Original escape
records are split into two four-value records. A selected-half metadata zero
with nonzero original metadata gets a new exact-BF16 escape record. Bad original
escape indices fail closed on both CPU and GPU.

For the released `embed.weight`, `[129280,5120]`, both transforms produced
identical complete packed and escape tensors. Its 152,751 original escape
records and zero selected-half collisions give:

| Representation | Packed bytes | Escape bytes | Resident total |
| --- | ---: | ---: | ---: |
| Canonical NF12 | 992,870,400 | 2,444,016 | 995,314,416 |
| Row-major NF12 | 992,870,400 | 2,444,016 | 995,314,416 |
| Dense BF16 | — | — | 1,323,827,200 |

Thus replacing the canonical layout for an embedding-only weight preserves
the **995,314,416-byte resident size**. In general, row-major needs eight
additional bytes per selected-half collision: the forced fixture has 254 and
grows from 3,104 to 5,136 bytes. A side-by-side canonical plus row-major cache
would add 995,314,416 GPU bytes on this released weight. The prototype itself
does not own such a cache.

## Full checkpoint load and conversion

The source is the released DeepSeek-V4.1-Flash-1017 `embed.weight` in
`model-00002-of-00048.safetensors`, SHA256
`aa1ac979edab508950a871c370fa645a875203a1a35c20ba9c1d45bb1ed46e3c`.
The source checkpoint stores dense BF16, so the measured path first encodes
canonical NF12 on GPU. Times are one end-to-end pass with the native binary
already compiled, a warm filesystem page cache, and synchronization around
GPU stages. They are load-stage observations, not steady-state kernel medians.

| Stage | Seconds |
| --- | ---: |
| Read indexed dense BF16 checkpoint tensor to CPU | 0.0387 |
| Dense BF16 CPU to GPU | 0.1215 |
| Canonical NF12 GPU encode | 0.0962 |
| Canonical to row-major on GPU | 0.0254 |
| **Dense checkpoint to GPU row-major via GPU transform** | **0.2818** |
| Canonical NF12 GPU to CPU, for CPU path | 0.3808 |
| Canonical to row-major on CPU | 5.1429 |
| CPU row-major to GPU | 0.0861 |
| **Dense checkpoint to GPU row-major via CPU transform** | **5.8661** |

The CPU path total includes the same checkpoint read, BF16 upload, and
canonical GPU encode. A checkpoint already storing canonical packed NF12
would skip those initial dense-source stages. The benchmark held only the
representations needed in each phase. The CUDA allocator was capped at 2.6%
of physical GPU1, about 2.68 GB; peak PyTorch allocated/reserved memory was
2,398,859,776 / 2,436,890,624 bytes. Production Lite was not modified or
restarted.

A second full-weight run measured 26.5 ms for GPU repacking and reproduced
the paired int64 code-ID results within 0.13 µs. During that run the largest
sampled `nvidia-smi` process allocation was 2,848 MiB (2.986 GB), below the
3 GB added-VRAM limit. Its read and encode stages varied, so the 0.2818 s
total above is an observed warm-cache pass rather than a startup guarantee.

## Runtime lookup on Lite code IDs

The code-ID fixture prefix was used at M1536/3072. Geometry was fixed from
the earlier GPU1 sweep: four lanes per 64-output block, L1-bypassing output
stores, `256` threads and `8` IDs per CTA at M1536; `128` threads and `2`
IDs per CTA at M3072. Eight-call CUDA graphs ran six paired ABBA rounds. Each
slash-separated column is its own paired phase; cache competition changes the
candidate timing between the canonical and dense phases.

| ID dtype | M | Canonical NF12 / row-major (µs) | Dense BF16 / row-major (µs) |
| --- | ---: | ---: | ---: |
| int64 | 1536 | 13.760 / 13.056 | 9.600 / 11.264 |
| int64 | 3072 | 26.880 / 22.656 | 18.624 / 19.968 |
| int32 | 1536 | 13.504 / 13.056 | 10.112 / 11.840 |
| int32 | 3072 | 26.304 / 22.784 | 19.200 / 20.544 |

All complete outputs matched independent dense BF16 lookup bit for bit,
including after CPU and GPU transforms. The separate `[16,128]` fixture
covered 254 sentinel collisions, two original escapes, signed zero,
subnormals, NaN payloads, invalid IDs, and both ID dtypes. Four targeted tests
also cover empty escape tables, corrupt original indices, repeated forwards,
and versioned checkpoint restoration. After repeated forwards the row-major
object retained exactly the packed and escape tensors above.

## Source and checkpoint lifecycle

The released checkpoint has distinct `embed.weight` and `head.weight` keys
(the latter in `model-00043-of-00048.safetensors`). A loader could convert
only `embed.weight` and release its canonical packed source after row-major
construction. Existing canonical NF12 checkpoint keys and operators cannot
reinterpret row-major bytes: a persistent row-major checkpoint needs its
versioned layout keys and explicit load selection. A model that ties an
embedding to a linear head still needs canonical NF12 for the head, or a
separately qualified row-major linear path; keeping both packed layouts costs
an additional 995,314,416 GPU bytes for this shape. No mutation-aware cache or
save/load route is installed in production by this prototype.

## Repeat4 coordination

Agent7's canonical fused repeat4 benchmark reports about 44.6 µs at M1536
and 79.5 µs at M3072 on code int64 IDs, with four BF16 activation copies.
The row-major single-copy gain suggests a possible improvement from its
contiguous vector stores and smaller selected-row reads. The four-copy output
is much larger, so the single-copy speedup cannot be carried over. A combined
kernel is **plausibly beneficial but unmeasured**; it should be evaluated in
the fused repeat4 branch, without duplicating that geometry sweep here. The
needed evidence is paired ABBA on the same physical GPU and full released
weight against canonical fused repeat4 and dense lookup plus repeat, at both
Lite M values, with exact four-copy BF16 bits, output-byte traffic, retained
layout bytes, and peak VRAM reported.

Prototype: [API](../gptqmodel_nf12/experimental_rowmajor.py),
[native transform and lookup](../gptqmodel_ext/nf12/experiments/rowmajor_embedding.cu),
[benchmark](../scripts/bench_nf12_rowmajor_layout_gpu1.py), and
[tests](../tests/test_nf12_rowmajor_experimental.py).
The [raw GPU1 report](nf12-rowmajor-layout-gpu1-20261111.json) contains all
timing samples, source hashes, storage counts, and memory measurements.
The [full-weight recheck](nf12-rowmajor-layout-gpu1-recheck-20261111.json)
records the second run.
