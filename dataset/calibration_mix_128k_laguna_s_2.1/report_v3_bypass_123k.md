# Laguna-S-2.1 coverage scan v3 — MoE routing bypass (completed run, 123,355 tokens)

First full scan with the MoE-aware scanner: expert-coverage profiling (`--moe-expert-coverage`)
plus routing bypass (`--moe-routing-bypass`), i.e. every router's `top_k` was raised from 10 to
256 during coverage so all experts received every token. This run predates the floor-fill stop
fix, so selection halted one shard short of the 131,072 dense floor; the follow-up run (report.md)
supersedes the final mix, but the per-shard numbers below are the reference record for the bypass
configuration.

## Configuration

| Setting | Value |
| --- | --- |
| Model | `/monster/data/model/Laguna-S-2.1-PER-LAYER` (219GB bf16, 48 layers, 256 experts, top-10) |
| Devices | 4x GPU (physical 0,1,2,3), `device_map=auto`, bf16 |
| Runtime | free-threaded CPython 3.14t, `PYTHON_GIL=0`, 32 greedy threads, 32 checkpoint I/O workers |
| Candidates | 14 shards (imatrix, nm_llm LLM, code/Magicoder, math/OpenMathInstruct-2, wiki zh/ru, pg19) |
| Target | `--target-tokens 131072`, mode `gain_per_token` |
| MoE | `--moe-expert-coverage` on, weight 1.0, min tokens 16; `--moe-routing-bypass` on (top_k 10 -> 256, 47 MoE layers, 12,032 experts); MoE floor auto-unset under bypass |
| Throughput | ~110-130 tok/s under bypass vs ~155 tok/s with normal routing (~1.2-1.4x cost, not the naive 25.6x) |

## Selected mix (greedy order)

| Step | Shard | Conditional gain | Score after | Cumulative tokens |
| --- | --- | --- | --- | --- |
| 1 | wiki_zh_00 | +1,539,866.76 | 55,525.25 | 29,673 |
| 2 | nm_llm_01 | +17,220.24 | 38,305.01 | 60,580 |
| 3 | nm_llm_03 | -2,060.64 (floor fill) | 40,365.65 | 91,322 |
| 4 | wiki_ru_00 | +358.44 | 40,007.21 | 123,355 |

- Starting score: 1,595,392.01 (the higher baseline vs prior layer-level runs reflects the
  MoE expert-coverage penalty term: an empty mix leaves all reference routed mass uncovered).
- Final score: 40,007.21; cumulative gain 1,555,384.80.
- Total: 95 rows / 123,355 tokens.
- Warning: `Target token floor 131072 not reached (reached 123355); candidate pool exhausted.`
  — caused by a lookahead stop-condition bug (fixed in the follow-up run), not actual pool
  exhaustion; 10 candidates remained.

## MoE expert coverage (bypass)

- 47 fused expert modules (`model.layers.{1..47}.mlp.experts`), 256 experts each = 12,032 experts.
- All 47 routers bypassed: top_k 10 -> 256, restored after scanning.
- Every expert received every token: min routed tokens per expert = 123,355 (== total dense
  tokens) for the selected mix; per-shard min == that shard's token count.
- Selected-mix uncovered routed mass: 0.0 (all reference-active experts covered).

## Per-shard standalone scores (lower = better tail alignment with reference)

| Shard | Tokens | Standalone score | Final conditional gain | Verdict |
| --- | --- | --- | --- | --- |
| wiki_zh_00 | 29,673 | 55,525.3 | +1,539,866.76 | selected |
| nm_llm_01 | 30,907 | 54,950.3 | +17,220.24 | selected |
| nm_llm_03 | 30,742 | 57,382.7 | -2,060.64 | selected (floor fill) |
| wiki_ru_00 | 32,033 | 63,267.6 | +358.44 | selected |
| code_00 | 31,927 | 63,922.7 | -701.81 | redundant |
| nm_llm_02 | 30,314 | 60,061.0 | -808.01 | redundant |
| code_01 | 32,624 | 64,904.8 | -1,097.77 | redundant |
| math_00 | 31,032 | 71,246.2 | -1,140.24 | redundant |
| nm_llm_00 | 31,041 | 60,643.0 | -1,162.39 | redundant |
| imatrix_01 | 31,743 | 57,773.6 | -1,575.53 | redundant |
| math_01 | 30,971 | 73,694.9 | -1,695.69 | redundant |
| imatrix_00 | 31,645 | 55,151.4 | -2,001.23 | redundant |
| pg19_01 | 31,359 | 68,013.2 | -4,183.47 | redundant |
| pg19_00 | 32,187 | 68,115.7 | -4,432.99 | redundant |

## Key observations

1. Under bypass + expert-coverage scoring, the selection changed materially vs the layer-level
   scan (which picked wiki_zh_00 + imatrix_00 at 61K): the MoE-aware objective un-saturated the
   search and pushed the mix to 4 shards / 123K tokens with mostly positive gains.
2. imatrix_00, the layer-level runner-up, became redundant once expert-coverage was scored —
   its expert activation profile overlaps wiki_zh_00's more than nm_llm's chat distribution does.
3. Bypass compute cost is modest (~1.2-1.4x) because at 1,024-token chunks the fused expert
   GEMMs already touch most experts; bypass mainly increases per-expert routed token mass.
4. pg19 (long-form books) is the least complementary source for this model under both objectives.
