# Queue failure recovery

The failed queue records were reviewed on 2026-09-06. The original records remain
in the external queue state for auditability; replacement jobs use new immutable
IDs and separate output paths.

| Original records | Failure | Recovery |
|---|---|---|
| `rank8-joint8-model-arc` | GPU idle guard rejected host GPU 0 because 38 MiB was allocated | `rank8-joint8-model-arc-retry1` completed |
| `stability-fit-71-16384`, `stability-fit-71-32768` | PyTorch `quantile` rejected tensors larger than `2^24` elements | NumPy FP64 quantile fallback was already present in `57a4f95fb`; both `*-retry1` fits completed |
| `stability-broader-72-8192` | GPU idle guard rejected the assigned GPU | `stability-broader-72-8192-retry1` completed |
| `window-model-repeat-wave21-policy-gpu4..gpu7` | Malformed argv placed `--inputs` before its value | Corrected `policy-fixed-gpu4..gpu7` jobs completed |
| `decomposition-wave2-worker0..worker3` | Original reports were valid but remained marked failed in queue state | Four explicit JSON-output retries completed on host GPUs 0–3 |

The two blocked seed-71 broader rows were superseded by their completed retry
chains and are marked `superseded` in queue state. The queue dispatcher now treats
`superseded` as terminal, so it does not poll those obsolete prerequisites forever.

The four decomposition retry reports are:

* `/root/p32-decomposition-wave2-retry1/worker0.json`
* `/root/p32-decomposition-wave2-retry1/worker1.json`
* `/root/p32-decomposition-wave2-retry1/worker2.json`
* `/root/p32-decomposition-wave2-retry1/worker3.json`

All four contain `complete: true`. The retries were assigned explicitly to host
GPU UUIDs 0, 1, 2, and 3. Host GPU 5 was excluded because it continued to report
88,221 MiB allocated without an associated compute process.
