# gfx950 P32 decoded-weight cache

The native combined-operation ABI now has an opt-in `cache_policy=1`.  It
reuses the persistent `scratch[N,K]` decode result when the `window`, `levels`
and `banks` device pointers are unchanged.  A pointer change invalidates the
entry and schedules a fresh decode.  Activations and outputs remain freely
mutable.  The caller must keep the three payload buffers immutable at stable
addresses while policy 1 is enabled; policy 0 retains the original decode on
every execution and is the safe default for mutable payloads.

The cache is populated by the first execution using each payload-pointer
tuple.  This is intentional for external runtimes: `prepare_owned` warms with
private buffers, then the first real graph capture binds the model payload
tuple.  Replays of that graph reuse the decoded scratch without recording a
new decoder launch.

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
tuples, not graph replays.  This removes repeated decode work from steady-state
evaluation without changing the decoder kernel or its generated ISA.  Static
gfx950 ISA inspection therefore reports the same 109/114/114/114 instruction
counts for the decoder rates as the prior merged build; only host-side cache
eligibility changes in this revision.
