# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Time one staged GSQ update at paper geometry and report peak VRAM.

The relaxed scalar weights are rebuilt for every microbatch because the pinned
author redraws Gumbel noise per microbatch. Hoisting them would change the
estimator, so this benchmark instead measures the scale-gradient reduction:
a contiguous group map lets the backward pass use a segment sum instead of a
collision-heavy ``scatter_add_``. Both variants keep identical noise cadence.

    python scripts/gsq_paper/bench_stage_update.py --device cuda:0
"""

import argparse
import sys
import time
from pathlib import Path

import torch

# Running as a script puts scripts/gsq_paper first on sys.path, so an installed
# gptqmodel would silently shadow this checkout and benchmark other code.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from gptqmodel.quantization import gsq_training as stage

# Llama-3.1-8B geometry at the paper's batch/microbatch shape.
DEFAULT_HIDDEN = 4096
DEFAULT_INTERMEDIATE = 14336
DEFAULT_GROUP = 128
DEFAULT_MICROBATCHES = 4
DEFAULT_TOKENS = 16


def _build(bits, device, *, group, hidden, intermediate, microbatches, tokens):
    choices = 4 if bits == 2 else 5
    weight = torch.randn(intermediate, hidden, device=device, dtype=torch.bfloat16)*.02
    scales = weight.abs().view(intermediate, -1, group).amax(2).clamp_min(1e-3)/2.5
    module = stage.GSQScalarTrainingModule(
        weight, scales, group, bits=bits,
        noise=torch.randn(choices, intermediate, hidden, device=device, dtype=torch.bfloat16),
    )
    optimizer = stage.GSQLion(
        module.optimizer_groups(assignment_lr=2e-4, scale_lr=1e-4, weight_decay=1.),
        betas=(.9, .95),
    )
    activations = torch.randn(tokens, hidden, device=device, dtype=torch.bfloat16)
    objective = lambda batch, weights: (activations@weights['weight'].T).float().pow(2).mean()
    microbatches = [(None, tokens)]*microbatches
    generator = torch.Generator(device=device).manual_seed(1)
    update = lambda: stage.train_stage_update(
        {'weight': module}, optimizer, microbatches, objective,
        generator=generator, temperature=1., multiplier=100.,
    )
    return module, update


def _measure(update, *, warmup=1, repeats=3):
    for _ in range(warmup):
        update()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    started = time.perf_counter()
    for _ in range(repeats):
        update()
    torch.cuda.synchronize()
    return (time.perf_counter()-started)/repeats*1000, torch.cuda.max_memory_allocated()/2**30


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--bits', type=int, nargs='+', default=[2, 3])
    parser.add_argument('--group', type=int, default=DEFAULT_GROUP)
    parser.add_argument('--hidden', type=int, default=DEFAULT_HIDDEN)
    parser.add_argument('--intermediate', type=int, default=DEFAULT_INTERMEDIATE)
    parser.add_argument('--microbatches', type=int, default=DEFAULT_MICROBATCHES)
    parser.add_argument('--tokens', type=int, default=DEFAULT_TOKENS)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('a CUDA device is required')
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    reference = stage.relaxed_scalar_weights

    def scatter_only(*parameters, **keywords):
        # Reproduce the pre-change call so both variants share noise cadence.
        keywords.pop('contiguous_group_size', None)
        return reference(*parameters, **keywords)

    print(f'{"bits":>4}  {"variant":<20} {"ms/update":>10} {"peak GiB":>9}')
    try:
        for bits in args.bits:
            for label, function in (('scatter_add', scatter_only), ('segment_sum', reference)):
                stage.relaxed_scalar_weights = function
                module, update = _build(
                    bits, device, group=args.group, hidden=args.hidden,
                    intermediate=args.intermediate, microbatches=args.microbatches,
                    tokens=args.tokens,
                )
                milliseconds, peak = _measure(update, repeats=args.repeats)
                print(f'{bits:>4}  {label:<20} {milliseconds:>10.1f} {peak:>9.2f}')
                del module, update
                torch.cuda.empty_cache()
    finally:
        stage.relaxed_scalar_weights = reference


if __name__ == '__main__':
    main()
