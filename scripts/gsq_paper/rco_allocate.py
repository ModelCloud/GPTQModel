# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Search a per-layer GSQ bit allocation at a paper budget with RCO.

GSQ keeps the scalar group-wise format after refinement, so blocks may mix
widths. The GSQ paper exploits exactly that to report non-uniform Llama rows at
2.37 and 2.62 bits per parameter, obtained with RCO (arXiv 2604.18556 Sec. 4.2,
Table 1, citing arXiv 2605.00649).

This driver builds the width database from real GSQ blocks (one width per
option), runs the projected Gumbel search on held-out calibration documents,
rounds the soft solution to an exact-budget assignment and writes both the
per-layer width map and the assignment's measured bits per parameter.

    python scripts/gsq_paper/rco_allocate.py \
        --model /monster/data/model/Llama-3.2-1B-Instruct \
        --corpus /models/datasets/gsq-bench/fineweb-paper \
        --out /monster/data/model/gsq-bench/rco-237 --target-bits 2.375
"""

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_from_disk

from gptqmodel.quantization.rco import (
    InterpolatedLlamaLayers,
    achieved_bits,
    projected_gumbel_search,
    round_with_budget,
)

PROJECTIONS = ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
               'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj')
PARAMETERIZED = 7


def documents(corpus, split, count):
    dataset = load_from_disk(str(Path(corpus) / split))
    if count > len(dataset):
        raise ValueError(f'{split} holds {len(dataset)} sequences, {count} requested')
    return [{'input_ids': dataset[index]['input_ids']} for index in range(count)]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', required=True)
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--target-bits', type=float, required=True,
                        help='paper budgets are 2.375 and 2.625 bits per parameter')
    parser.add_argument('--options', default='2,3',
                        help='comma-separated widths the search may choose per block')
    parser.add_argument('--arms', required=True,
                        help='comma-separated staged checkpoint arms, one per width, '
                             'using {bits} as the width placeholder')
    parser.add_argument('--layers', type=int, default=16)
    parser.add_argument('--group-size', type=int, default=128)
    parser.add_argument('--gptq-seqs', type=int, default=512)
    parser.add_argument('--calib-seqs', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=4,
                        help='sequences per forward pass; matches the reference default')
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


def load_block(model, checkpoint, width):
    """Rebuild one packed block from a staged checkpoint and return its widths.

    Staged GSQ writes one checkpoint per block, so an arm already on disk is
    the width database entry for that block: no re-quantization is required.
    """
    from gptqmodel.looper.gsq_training_model import _restore_packed_layer

    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if saved.get('layer_index') is None:
        raise ValueError(f'Block checkpoint lacks a layer index: {checkpoint}')
    bits = saved['signature'].get('bits')
    if bits != width:
        raise ValueError(f'{checkpoint} holds W{bits}, expected W{width}')
    layer = model.model.layers[saved['layer_index']]
    return _restore_packed_layer(layer, saved['state_dict'], names=PROJECTIONS, bits=width,
                                 group_size=group_size_of(saved), device='cpu')


def group_size_of(saved):
    return saved['signature']['group_size']


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    options = tuple(sorted(int(value) for value in args.options.split(',')))
    if len(options) < 2:
        raise SystemExit('RCO needs at least two width options')
    templates = [value for value in args.arms.split(',') if value]
    if len(templates) == 1 and '{bits}' in templates[0]:
        templates = templates * len(options)
    if len(templates) != len(options):
        raise SystemExit(f'{len(options)} widths need {len(options)} arms or one {{bits}} template, '
                         f'got {len(templates)}')
    arms = [Path(value.format(bits=width)) for value, width in zip(templates, options)]
    for arm in arms:
        if not arm.exists():
            raise SystemExit(f'Width database arm is missing: {arm}')

    from gptqmodel import GPTQModel
    from gptqmodel.models._const import DEVICE
    from gptqmodel.quantization import FORMAT, GPTQConfig

    config = GPTQConfig(bits=options[-1], group_size=args.group_size, sym=True, desc_act=False,
                        act_group_aware=False, format=FORMAT.GPTQ_V2, device=DEVICE.CUDA,
                        offload_to_disk=False)
    wrapper = GPTQModel.load(args.model, quantize_config=config, dtype=torch.bfloat16,
                            attn_implementation='sdpa')
    wrapper.model.to(args.device).eval()

    calib_docs = documents(args.corpus, 'validation', args.calib_seqs)

    database, param_counts = {}, {}
    started = time.monotonic()
    for index in range(args.layers):
        for width, arm in zip(options, arms):
            checkpoint = arm / f'layer-{index:02d}.pt'
            if not checkpoint.exists():
                raise SystemExit(f'Width database is missing block {index} at W{width}: {checkpoint}')
            packed = load_block(wrapper.model, checkpoint, width)
            for name in PROJECTIONS:
                key = f'model.layers.{index}.{name}'
                # ``dequantize_weight`` returns [in, out]; nn.Linear holds
                # [out, in]. Square projections hide the difference, so this
                # transpose is required for k/v/gate/up/down.
                weight = packed.get_submodule(name).dequantize_weight().T
                live = wrapper.model.get_submodule(key).weight
                if weight.shape != live.shape:
                    raise ValueError(f'{key} rebuilt as {tuple(weight.shape)}, '
                                     f'live module is {tuple(live.shape)}')
                database.setdefault(key, {})[width] = weight.detach().to(torch.float32).cpu()
                param_counts[key] = int(weight.numel())
            del packed
        print(f'  loaded block {index+1}/{args.layers} widths={options} '
              f'({time.monotonic()-started:.0f}s)', flush=True)

    names = sorted(database)
    layer_weights = {name: [database[name][width] for width in options] for name in names}
    costs = torch.tensor([float(width) for width in options])

    # Search over decoder blocks, not individual projections. Staged GSQ trains
    # and exports one width per block, so a per-projection assignment could not
    # be materialized without changing the budget; grouping by block makes the
    # searched budget exactly the one the checkpoint achieves.
    blocks = sorted({int(name.split('.')[2]) for name in names})
    groups = [[name for name in names if int(name.split('.')[2]) == block]
              for block in blocks]
    block_params = [sum(param_counts[name] for name in group) for group in groups]
    weights = torch.tensor([float(count) for count in block_params])
    print('database ready:', len(names), 'modules in', len(groups), 'blocks,',
          int(weights.sum().item()), 'parameters')

    interpolated = InterpolatedLlamaLayers(wrapper.model, layer_weights, weights, costs,
                                           control_device=args.device,
                                           groups=groups).setup()
    # Batch documents into sequences of equal length. The corpus splits every
    # document to a fixed length, so stacking is lossless and the per-step cost
    # drops by the batch factor.
    stacked = [torch.tensor(document['input_ids'], dtype=torch.long) for document in calib_docs]
    lengths = {int(ids.numel()) for ids in stacked}
    if len(lengths) != 1:
        raise SystemExit(f'calibration documents must share one length, saw {sorted(lengths)}')
    stacked = torch.stack(stacked)
    text = [chunk.to(args.device) for chunk in stacked.split(args.batch_size)]
    print(f'calibration: {len(calib_docs)} documents of {lengths.pop()} tokens '
          f'in {len(text)} batches of {args.batch_size}', flush=True)

    def objective(ids):
        return wrapper.model(input_ids=ids, labels=ids).loss

    try:
        interpolated.retract(args.target_bits)
        probs, history = projected_gumbel_search(
            interpolated, text, args.target_bits, steps=args.steps, lr=args.lr,
            objective=objective, gumbel_samples=1, log_every=10)
    finally:
        interpolated.cleanup()

    # Persist the search result before rounding: the search is the expensive
    # part, and a failure in the cheap tail must not force another run.
    torch.save({'probs': probs.detach().cpu(), 'history': history,
                'weights': weights.cpu(), 'costs': costs.cpu(), 'names': names,
                'target_bits': args.target_bits}, out / 'probs.pt')

    assignment = round_with_budget(probs, weights, costs, args.target_bits)
    exact = achieved_bits(assignment, weights, costs)

    # One width per decoder block: the unit staged GSQ trains. The assignment
    # is already at block granularity, so no reduction (and no budget drift).
    layer_bits = {block: options[option] for block, option in zip(blocks, assignment.tolist())}
    block_bits = exact

    result = {
        'model': args.model,
        'target_bits': args.target_bits,
        'options': list(options),
        'layer_bits': {str(index): layer_bits[index] for index in sorted(layer_bits)},
        'projection_count': len(names),
        'group_count': len(groups),
        'steps': args.steps,
        'drift_max': max(record['drift'] for record in history),
        'drift_final': history[-1]['drift'],
        'projection_bits_per_parameter': exact,
        'block_bits_per_parameter': block_bits,
        'width_counts': {f'W{width}': sum(1 for v in layer_bits.values() if v == width)
                         for width in options},
        'search_seconds': None,
        'history': history,
    }
    (out / 'assignment.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({key: result[key] for key in
                      ('target_bits', 'layer_bits', 'width_counts',
                       'projection_bits_per_parameter', 'block_bits_per_parameter',
                       'drift_max', 'drift_final')}, indent=2))


if __name__ == '__main__':
    main()
