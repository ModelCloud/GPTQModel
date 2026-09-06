# User-approved inference gate update

The user raised the localized inference mean absolute error limit from 0.002 to
0.003 (inclusive). Maximum absolute error remains <=0.046875; non-finite outputs
still fail. This changes acceptance policy, not checkpoint bits or kernel arithmetic.

Reassessing the 108 saved layer cases gives **108/108 passes for both planar and
window**. Layer 1 MLP down-projection at M=1 now passes: window MAE
0.0027303189164769037, max 0.01091650128364563; planar MAE
0.0027302544096556858, max 0.0109233558177948.

The local error gate block is lifted. Full-model performance and held-out quality
validation remain required for promotion. The original measurement files retain
their historical 0.002 decisions; [reassessment](results/gate-reassessment.json)
records the new decisions without rerunning or altering measured errors.
