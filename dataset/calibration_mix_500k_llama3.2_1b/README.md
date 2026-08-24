# Llama 3.2 1B calibration coverage mix

`generate.py` downloads candidate datasets locally, excludes the locked benchmark and YAQA rows by canonical content hash, profiles every Llama linear-module input group, and assembles a calibration mix.

The configured desired target is 500,000 tokens and the minimum acceptable target is 256,000 tokens. Positive conditional-gain shards are selected first. If they do not reach the minimum, the least-redundant remaining shards are added only until the minimum is met.

The recorded run selected five shards with 322,161 tokenizer-counted tokens and 302,011 scanner-effective tokens. All content-hash intersections between the selected calibration data, held-out coverage reference, locked YAQA slice, and benchmark slice were zero.

For QVQ YAQA runs, ordinary calibration is lifecycle-forward-only: module input/output Hessians come from the separate YAQA Sketch-B stream. Optimizing this mix therefore does not alter YAQA weights unless a calibration-dependent replay or alignment control is enabled.

Using the full mix as the disjoint YAQA stream for Llama 3.2 1B W2 V2B2-P32 changed the checkpoint and improved the locked 300-row diagnostics from 0.324339 to 0.264975 final KL and from 71.4331% to 76.0206% legacy teacher-forced 32-position top-1 agreement. That historical number predates the independent greedy-rollout Divergence-300 @32 protocol and must not be compared with its trajectory-survival score. The ordinary lifecycle stream, optimized YAQA stream, and evaluation slice had zero canonical content-hash intersections.
