# Matched W2.5 prepared-Fisher search, F6 seed 7

Run `artifacts/gsq-p32/matched-fisher-w25-seed7-v2` completed at source
`3c402a6e4`. Real Llama 3.2 1B block-0 Q/K/V weights use 16 calibration
documents (3,767 tokens) and 32 disjoint held-out documents (6,367 tokens).
All other projections retain the F6 snapshot. Full-model propagation uses the
FP32 canonical reference; native checks cover the selected layers only.

The baseline is fresh W2.5 YAQA, not the original F6 model. GSQ uses 100 Adam
steps, seed 7 and 33 fixed whole-tile candidates. Deterministic coordinate
descent uses three sweeps on identical saved candidates, teacher and Fisher
inputs. This matches search space and objective, not compute budget. Shared
artifact hashes and per-projection histories are in the report.

| Arm | Final-logit KL | MSE | Top-1 | Top-5 | Top-10 |
|---|---:|---:|---:|---:|---:|
| W2.5 YAQA baseline | 0.1101865335 | 0.4462685826 | 0.8603738024 | 0.8304067850 | 0.8278152976 |
| GSQ | 0.1101865335 | 0.4462685826 | 0.8603738024 | 0.8304067850 | 0.8278152976 |
| Deterministic | 0.1101295253 | 0.4461068823 | 0.8555049474 | 0.8315376158 | 0.8277367677 |

These aggregate metrics are token-weighted. Reported bootstrap intervals instead
use paired document means; the estimands differ. For deterministic minus
baseline, document-mean 95% intervals are KL [-0.00010644, -0.00003574],
MSE [-0.00034405, -0.00000865], and Top-1 [-0.0077352, -0.0049000].
Thus the small KL/MSE improvements and Top-1 regression are clear under this
protocol. Top-5 improves; Top-10 is noise-consistent. No promotion is justified.

GSQ changes zero tiles. Deterministic changes Q/K/V by 26/16/9 tiles, lowering
prepared-Fisher objectives by approximately 1.58%/3.25%/0.70%. An independent
payload audit verifies baseline pool identity, each exported tile's membership
in the shared pool, serialized-file hashes and GSQ baseline equality. QKV
payload words total 1,966,080 bytes in each arm (excludes scale/bank metadata).

All nine selected-layer native reload checks pass independent mean/max gates
0.002/0.046875. Worst observed mean is 0.001141615; worst max is 0.025087357.
Physical GPU 0: PG506-230, SM80, UUID
`GPU-737e2423-874a-23a4-1126-dfbe3e77c294`, PCI DE:00.0, 124 SMs.
Three idle preflight samples passed; the lease was released after completion.

The first run is preserved separately: an FP32 evaluation-order difference
triggered the baseline-agreement gate. Version 2 uses GSQ's operation order,
with the original tolerance. The completed run demonstrates that deterministic
search can find improving calibration candidates in this frozen pool while this
GSQ setup retains baseline. It does not establish full-model recovery, a paper
reproduction, a uniform W2.5 model, or a full native-model export qualification.

Artifacts: [report](../../artifacts/gsq-p32/matched-fisher-w25-seed7-v2/report.json),
[payload audit](../../artifacts/gsq-p32/matched-fisher-w25-seed7-v2/payload-audit.json).
Reproduce with `scripts.validate_qvq_gsq_layers --gsq-lifecycle
--compare-deterministic --target-bits 2.5 --output NEW_RUN`, preparing first with
`--prepare`, then executing under an exclusive GPU allocator lease.
