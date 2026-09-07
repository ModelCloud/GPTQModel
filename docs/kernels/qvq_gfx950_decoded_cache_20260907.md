# gfx950 P32 decoded-weight cache

The native combined-operation ABI now has an opt-in `cache_policy=1`.  It
reuses the persistent `scratch[N,K]` decode result when the `window`, `levels`
and `banks` device pointers are unchanged.  A pointer change invalidates the
entry and schedules a fresh decode.  Activations and outputs remain freely
mutable.  The caller must keep the three payload buffers immutable at stable
addresses while policy 1 is enabled; policy 0 retains the original decode on
every execution and is the safe default for mutable payloads.

The cache is populated by the first execution using each payload-pointer
tuple. Eager calls with a stable model tuple then reuse decoded scratch. For
graph capture, the decision is made while recording: if the captured tuple was
already prewarmed with the same pointers, the graph contains only the GEMM;
otherwise the first capture records a decode and every replay repeats that
recorded decode. ZML's current `prepare_owned` path warms private buffers, so
its model graph needs a future payload-preparation hook to realize the full
graph-replay benefit.

## Validation and profile

On MI355X (gfx950), the combined native GPU test passed with both policies:

```text
QVQ_GFX950_CACHE_POLICY=0: 3 passed
QVQ_GFX950_CACHE_POLICY=1: 3 passed
```

The policy-0 test mutates packed words, LUT, banks and activations across graph
replays; policy 1 replays an immutable payload tuple and checks the same FP64
reference and MAE/max gates.  A rocprofv3 kernel trace counted decoder
dispatches across the five rates:

| rate | policy 0 | policy 1 |
|---:|---:|---:|
| 4 | 7 | 1 |
| 5 | 7 | 1 |
| 6 | 8 | 5 |
| 7 | 8 | 5 |
| 8 | 8 | 5 |

The remaining policy-1 launches are the distinct preparation/capture payload
tuples. This profile uses a direct ABI plan prewarmed with the model pointers,
so it demonstrates the best-case graph result; ZML's private-buffer warmup is
the conservative case described above. The cache changes host-side decode
eligibility only and does not change the decoder kernel or generated ISA.
Static gfx950 ISA inspection therefore reports the same
109/114/114/114/109 instruction counts as the prior merged build.
