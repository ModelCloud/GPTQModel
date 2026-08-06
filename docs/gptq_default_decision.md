# GPTQ accuracy-first default decision

## Outcome

GPTQModel Ultra retains this general GPTQ default profile:

```text
+----------------------+------------------------------------+
| Region               | Default                            |
+----------------------+------------------------------------+
| Hessian damping      | Fixed 5% GPTQ damping              |
| Range selection      | Activation-aware scale search      |
| Adaptive damping     | Disabled unless explicitly enabled |
| GPTQ-error clipping  | Disabled unless explicitly enabled |
+----------------------+------------------------------------+
```

The selected profile is exported in code as `GPTQ_DEFAULT_DAMP_PERCENT`,
`GPTQ_DEFAULT_DAMP_AUTO_INCREMENT`, and `GPTQ_DEFAULT_SCALE_SEARCH`. The
adaptive features retain PR 215's explicit opt-in contract.

## Decision rule

Post-quantization error is the primary criterion. Runtime and VRAM are only
tie-breakers after numerical validity, task non-inferiority, direct held-out
output error, and KLD.

```text
finite and canonical GPTQ math?
|
+-- no --> reject
|
+-- yes --> any task metric worse by more than 0.5% relative?
            |
            +-- yes --> do not make it the general default
            |
            +-- no --> lower held-out MAE/RMSE/relative-L2 and KLD?
                        |
                        +-- clear winner --> select it
                        |
                        +-- mixed --> prefer lower distributional error and
                                      stronger task/top-1 robustness
                                      |
                                      +-- still tied --> use runtime, VRAM,
                                                         simplicity, and compatibility
```

## Qwen3-8B task matrix

Every cell starts from the same dense BF16 `/monster/data/model/Qwen3-8B`
checkpoint and performs a fresh 4-bit, symmetric, group-128 quantization. The
calibration set contains 128 samples concatenated to 512 tokens. Evaluation
uses all 1,209 GSM8K Platinum samples and all 1,172 ARC Challenge samples.
The wall time includes quantization, save/reload, and post-quant evaluation.

```text
+------+------------------+-------------------------+------------+----------+----------+---------+-----------+
| Cell | Damping          | Range selection         | GSM8K acc  | ARC acc  | ARC norm | Wall s  | vs A note |
+------+------------------+-------------------------+------------+----------+----------+---------+-----------+
| A    | Fixed GPTQ 5%    | Activation scale search | 0.91480562 | 0.541809 | 0.554608 | 686.4   | selected  |
| B    | Fixed GPTQ 5%    | GPTQ-error clipping     | 0.89826303 | 0.530717 | 0.540102 | 822.1   | loses all |
| C    | Adaptive v4.2    | Activation scale search | 0.90488007 | 0.543515 | 0.557167 | 902.6   | mixed     |
| D    | Adaptive v4.2    | GPTQ-error clipping     | 0.90984285 | 0.543515 | 0.547782 | 1085.9  | mixed     |
+------+------------------+-------------------------+------------+----------+----------+---------+-----------+
```

Relative to A:

```text
+------+-------------+-----------+----------------+-------------+
| Cell | GSM8K delta | ARC delta | ARC norm delta | Wall delta  |
+------+-------------+-----------+----------------+-------------+
| B    | -1.808%     | -2.047%   | -2.615%        | +19.8%      |
| C    | -1.085%     | +0.315%   | +0.462%        | +31.5%      |
| D    | -0.542%     | +0.315%   | -1.231%        | +58.2%      |
+------+-------------+-----------+----------------+-------------+
```

## Disjoint held-out output-error matrix

The deterministic GPU 7 micro-test uses BF16 weights, 4 bits, group size 128,
two groups per projection, 256 calibration rows, and 256 independently seeded
held-out rows. It covers three Qwen-like projection roles and six geometries:
Gaussian, weight outlier, activation outlier, correlated, rank deficient, and
ill conditioned (18 cases per cell). The adaptive cells exercise module priors
and prior-group online feedback. KLD is a softmax sensitivity diagnostic for
hidden projections, not perplexity.

Environment: NVIDIA PG506-230 (`sm_80`), torch 2.13.0+cu130, CUDA 13.0,
Triton 3.7.1. The process starts with
`PYTORCH_ALLOC_CONF=expandable_segments:True,max_split_size_mb:1024,garbage_collection_threshold:0.50`.

```text
+------+------------------+-------------------------+-----------+-----------+-------------+-----------+----------+
| Cell | Damping          | Range selection         | Mean MAE  | Mean RMSE | Mean rel-L2 | Mean KLD  | Top-1    |
+------+------------------+-------------------------+-----------+-----------+-------------+-----------+----------+
| A    | Fixed GPTQ 5%    | Activation scale search | 40.909401 | 53.345080 | 0.12108388  | 8.9382463 | 0.865234 |
| B    | Fixed GPTQ 5%    | GPTQ-error clipping     | 53.424039 | 68.254309 | 0.13728161  | 11.285324 | 0.836589 |
| C    | Adaptive v4.2    | Activation scale search | 40.651270 | 53.019802 | 0.11856639  | 9.6723653 | 0.862630 |
| D    | Adaptive v4.2    | GPTQ-error clipping     | 52.624468 | 67.302017 | 0.13439534  | 10.126013 | 0.845269 |
+------+------------------+-------------------------+-----------+-----------+-------------+-----------+----------+
```

Relative to A, adaptive damping plus activation scale search (C) improves mean
MAE by 0.631%, RMSE by 0.610%, and relative-L2 by 2.079%, but worsens mean KLD
by 8.213%, lowers held-out top-1 agreement by 0.30%, and loses 1.085% GSM8K.
That is a real tradeoff, not a universal post-quant error win. Its ARC gains
remain inside the 0.5% non-inferiority band. Fixed damping therefore provides
the more robust general default while adaptive damping remains available for
model-specific tuning.

Both GPTQ-error-clipping cells are materially worse than their activation
scale-search counterparts in the aggregate held-out error matrix. Cell B is
also dominated by A on every model task and costs 19.8% more wall time. Exact
GPTQ-error clipping remains useful as an explicit research/tuning objective,
but these data do not justify enabling it globally.

## Scope

This decision is evidence-driven, not a claim that one Qwen3-8B calibration
sample covers every architecture, bit width, group size, or domain. A future
default change should rerun the same fail-closed decision tree on multiple
model families and require direct output-error/KLD non-inferiority as well as
task-level non-inferiority. The opt-in controls make that experimentation
possible without silently changing existing GPTQ checkpoints.
