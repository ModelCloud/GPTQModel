# QVQ-ZML coverage matrix

Copy one row per kernel family or specialization into the paired PRs. Never use a blank disposition.

| QVQ family / symbol | ABI + algorithm | SM | dtype / layout | M | K | N | bits | grouped | split + BM/BN/BK/stages/warps | workspace | ZML policy/admission | StableHLO/XLA attrs + validation | cache identity | graph replay | runtime proof | numerical gate | performance gate | disposition | owner + issue/reason |
|---|---|---|---|---:|---:|---:|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Example | vN / N | sm90 | f16/P32 | 1-16 | multiple of 256 | multiple of 256 | 4-7 | no | ordered split | queried bytes | function:line | attributes/predicate | fields/version | pass | counter + kernel name | FP32/FP64 result | warmed delta | wired | PR link |

Required summary:

- QVQ source commit:
- ZML pinned QVQ commit:
- Relevant-payload comparison: exact / equivalent / differs
- Newly exported families:
- ZML-only algorithms checked:
- Optimized launches observed:
- Portable fallbacks observed:
- Intentionally unwired items and owners:
