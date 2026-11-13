# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Search block or projection bit allocations from staged GSQ/GPTQ arms.

The default paper preset uses dense-teacher KL, independent projection decisions,
W2/W3/W4 candidates and four Gumbel draws per update. Use --preset legacy for
cross-entropy, block decisions, W2/W3 and one draw. Both use the supplied arms;
this does not construct the paper's full W2-W8 GPTQ database automatically.
"""

import argparse
import gc
import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import torch


DEFAULT_BATCH_SIZE = 4
DEFAULT_BATCH_TOKENS = 2048


def resolve_batch_size(requested, backend, sequence_length, batch_tokens):
    """Choose how many sequences share one forward pass.

    Torch keeps the reference default of four sequences per forward. The native
    MLX search additionally materializes every interpolated candidate weight on
    each forward, and that work is proportional to the parameter count rather
    than the token count, so a short forward spends most of its time rebuilding
    weights that a wider forward would amortize. Sizing the forward by a token
    budget instead keeps that overhead small. It does not change the objective:
    the per-batch fractions are token-weighted, so any partition of the
    calibration set yields the same token mean.
    """
    if requested is not None:
        return requested
    if backend != 'mlx':
        return DEFAULT_BATCH_SIZE
    return max(1, batch_tokens // max(1, sequence_length))


def documents(corpus, split, count):
    from datasets import load_from_disk

    dataset = load_from_disk(str(Path(corpus) / split))
    if count > len(dataset):
        raise ValueError(f'{split} holds {len(dataset)} sequences, {count} requested')
    return [{'input_ids': dataset[index]['input_ids']} for index in range(count)]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', required=True)
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--target-bits', type=float, required=True,
                        help='paper budgets are 2.375 and 2.625 bits per parameter')
    parser.add_argument('--preset', choices=('legacy', 'paper'), default='paper',
                        help='allocation configuration (default: paper)')
    parser.add_argument('--objective', choices=('ce', 'kl'))
    parser.add_argument('--grouping', choices=('block', 'projection'))
    parser.add_argument('--gumbel-samples', type=int)
    parser.add_argument('--kl-chunk-tokens', type=int, default=128)
    parser.add_argument('--teacher-cache', choices=('device', 'disk'), default='device',
                        help='retain teacher hidden states on the search device or stream them from disk')
    parser.add_argument('--options', help='ordered, comma-separated staged widths (2, 3, 4)')
    parser.add_argument('--arms', required=True,
                        help='comma-separated staged checkpoint arms, one per width, '
                             'using {bits} as the width placeholder')
    parser.add_argument('--layers', type=int, help='must cover all decoder blocks; inferred by default')
    parser.add_argument('--group-size', type=int, default=128)
    parser.add_argument('--gptq-seqs', type=int, default=512)
    parser.add_argument('--calib-seqs', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=None,
                        help='sequences per forward pass; defaults to 4 on Torch and to the '
                             'largest count that fits --batch-tokens on MLX')
    parser.add_argument('--batch-tokens', type=int, default=DEFAULT_BATCH_TOKENS,
                        help='token budget used to size the MLX forward pass '
                             f'(default: {DEFAULT_BATCH_TOKENS})')
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--device', default='auto', help='Torch device used to reconstruct staged arms')
    parser.add_argument('--backend', choices=('auto', 'torch', 'mlx'), default='auto',
                        help='differentiable search backend; auto prefers MLX on Apple silicon')
    args = parser.parse_args(argv)
    paper = args.preset == 'paper'
    args.objective = args.objective or ('kl' if paper else 'ce')
    args.grouping = args.grouping or ('projection' if paper else 'block')
    args.gumbel_samples = args.gumbel_samples if args.gumbel_samples is not None else (4 if paper else 1)
    args.options = args.options or ('2,3,4' if paper else '2,3')
    for name in ('calib_seqs', 'batch_tokens', 'steps', 'gumbel_samples', 'kl_chunk_tokens'):
        if getattr(args, name) < 1:
            parser.error(f'--{name.replace("_", "-")} must be positive')
    if args.batch_size is not None and args.batch_size < 1:
        parser.error('--batch-size must be positive')
    return args


def allocation_groups(param_counts, grouping, *, blocks=None):
    """Partition decisions while retaining each projection's parameter weight."""
    names = sorted(param_counts)
    if grouping == 'projection':
        groups = [[name] for name in names]
    elif grouping == 'block':
        if blocks is None:
            raise ValueError('Block grouping requires the model definition inventory')
        groups = [[name for name in block.names if name in param_counts] for block in blocks]
        groups = [group for group in groups if group]
        if sorted(name for group in groups for name in group) != names:
            raise ValueError('RCO block inventory must cover every allocated projection exactly')
    else:
        raise ValueError(f'Unknown RCO grouping: {grouping}')
    return groups, torch.tensor([sum(param_counts[name] for name in group) for group in groups],
                                 dtype=torch.float32)


def load_block(model, checkpoint, width, *, block, group_size):
    """Rebuild one packed block from a staged checkpoint and return its widths.

    Staged GSQ writes one checkpoint per block, so an arm already on disk is
    the width database entry for that block: no re-quantization is required.
    """
    from gptqmodel.looper.gsq_training_model import _restore_packed_layer

    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if saved.get('layer_index') != block.index or group_size_of(saved) != group_size:
        raise ValueError(f'Block index or group size differs in {checkpoint}')
    bits = saved['signature'].get('bits')
    if bits != width:
        raise ValueError(f'{checkpoint} holds W{bits}, expected W{width}')
    layer = model.get_submodule(block.path)
    # Packed checkpoints have qweight/qzeros/scales, not dense .weight entries.
    # Restore the entire declared inventory; strict loading rejects partial arms.
    return _restore_packed_layer(layer, saved['state_dict'], names=block.projections, bits=width,
                                 group_size=group_size_of(saved), device='cpu')


def group_size_of(saved):
    return saved['signature']['group_size']


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    options = tuple(int(value) for value in args.options.split(','))
    if len(options) < 2 or options != tuple(sorted(set(options))) or not set(options) <= {2, 3, 4}:
        raise SystemExit('RCO staged arms require distinct increasing widths from 2,3,4')
    if not options[0] < args.target_bits < options[-1]:
        raise SystemExit('RCO target must lie strictly between the lowest and highest options')
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
    from gptqmodel.quantization import FORMAT, GPTQConfig
    from gptqmodel.quantization.rco import (
        InterpolatedLlamaLayers,
        LlamaTeacherKL,
        achieved_bits,
        budget_constrained_argmax,
        discover_blocks,
        projected_gumbel_search,
    )
    from gptqmodel.quantization.rco_mlx import (
        rco_mlx_available,
        rco_mlx_supports_model,
    )
    from gptqmodel.utils.torch import resolve_torch_device

    device = resolve_torch_device(args.device)
    args.device = str(device)
    args.backend = ('mlx' if rco_mlx_available() else 'torch') if args.backend == 'auto' else args.backend
    if args.backend == 'mlx' and not rco_mlx_available():
        raise SystemExit('MLX RCO requires macOS arm64 with `pip install "gptqmodel[mlx]"`')
    config = GPTQConfig(bits=options[-1], group_size=args.group_size, sym=True, desc_act=False,
                        act_group_aware=False, format=FORMAT.GPTQ_V2, device=device,
                        offload_to_disk=False)
    wrapper = GPTQModel.load(args.model, quantize_config=config, dtype=torch.bfloat16,
                            attn_implementation='sdpa')
    wrapper.model.to(device).eval().requires_grad_(False)
    blocks = discover_blocks(wrapper)
    layers = len(blocks)
    if args.layers is not None and args.layers != layers:
        raise SystemExit('--layers must cover the complete model for an exportable allocation')
    args.layers = layers

    calib_docs = documents(args.corpus, 'validation', args.calib_seqs)

    database, param_counts = {}, {}
    started = time.monotonic()
    for block in blocks:
        index = block.index
        for width, arm in zip(options, arms):
            checkpoint = arm / f'layer-{index:02d}.pt'
            if not checkpoint.exists():
                raise SystemExit(f'Width database is missing block {index} at W{width}: {checkpoint}')
            packed = load_block(wrapper.model, checkpoint, width, block=block, group_size=args.group_size)
            for name, key in zip(block.projections, block.names):
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

    groups, weights = allocation_groups(param_counts, args.grouping, blocks=blocks)
    print('database ready:', len(names), 'modules in', len(groups), args.grouping, 'groups,',
          int(weights.sum().item()), 'parameters')

    # Batch documents into sequences of equal length. The corpus splits every
    # document to a fixed length, so stacking is lossless and the per-step cost
    # drops by the batch factor.
    stacked = [torch.tensor(document['input_ids'], dtype=torch.long) for document in calib_docs]
    lengths = {int(ids.numel()) for ids in stacked}
    if len(lengths) != 1:
        raise SystemExit(f'calibration documents must share one length, saw {sorted(lengths)}')
    sequence_length = lengths.pop()
    args.batch_size = resolve_batch_size(
        args.batch_size, args.backend, sequence_length, args.batch_tokens
    )
    stacked = torch.stack(stacked)
    text = [chunk.to(device) for chunk in stacked.split(args.batch_size)]
    print(f'calibration: {len(calib_docs)} documents of {sequence_length} tokens '
          f'in {len(text)} batches of {args.batch_size}', flush=True)

    # Cache the dense teacher BEFORE setup replaces the live projections.
    with TemporaryDirectory(prefix='rco-teacher-', dir=out) as cache:
        if args.backend == 'mlx':
            import mlx.core as mx
            from mlx_lm.utils import load as load_mlx

            from gptqmodel.quantization.rco_mlx import (
                MLXCausalLMLoss,
                MLXInterpolatedLlamaLayers,
                MLXTeacherKL,
                projected_gumbel_search_mlx,
                write_mlx_assignment,
            )

            # Releasing the reconstruction model before loading MLX avoids
            # keeping two full dense decoder graphs resident in unified memory.
            # The packed candidates have already been copied to layer_weights.
            mlx_batches = [batch.detach().cpu().numpy() for batch in text]
            del text, database, wrapper
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            elif device.type == 'mps':
                torch.mps.empty_cache()

            model, _, config = load_mlx(args.model, return_config=True, trust_remote_code=True)
            if not rco_mlx_supports_model(config.get('model_type')):
                raise SystemExit(
                    'MLX RCO supports dense Llama and Qwen3 models, got '
                    f'{config.get("model_type")}'
                )
            model.eval()
            objective = (MLXTeacherKL(model, mlx_batches, cache, chunk_tokens=args.kl_chunk_tokens,
                                      resident_cache=args.teacher_cache == 'device')
                         if args.objective == 'kl' else MLXCausalLMLoss(model, mlx_batches))
            try:
                interpolated = MLXInterpolatedLlamaLayers(
                    model, layer_weights, weights, costs, groups=groups)
                del layer_weights
                mx.random.seed(args.seed)
                search_started = time.monotonic()
                probs, history = projected_gumbel_search_mlx(
                    interpolated, list(range(len(mlx_batches))), args.target_bits,
                    steps=args.steps, lr=args.lr, objective=objective,
                    gumbel_samples=args.gumbel_samples, log_every=10,
                    batch_weights=[ids.size for ids in mlx_batches])
                mlx_probabilities = probs
                mlx_logits = interpolated.alpha
                mlx_weights = weights.detach().cpu().numpy()
                mlx_costs = costs.detach().cpu().numpy()
            finally:
                objective.cleanup()
                if 'interpolated' in locals():
                    interpolated.cleanup()
                    del interpolated
            del objective, model, mlx_batches
            search_seconds = time.monotonic() - search_started
            saved = write_mlx_assignment(
                out / 'probs.pt', probabilities=mlx_probabilities,
                logits=mlx_logits, history=history, weights=mlx_weights,
                costs=mlx_costs, names=names, groups=groups,
                target_bits=args.target_bits)
            probs, logits = saved['probs'], saved['logits']
        else:
            objective = (LlamaTeacherKL(wrapper.model, text, cache, chunk_tokens=args.kl_chunk_tokens,
                                         resident_cache=args.teacher_cache == 'device')
                         if args.objective == 'kl' else
                         lambda index: wrapper.model(input_ids=text[index], labels=text[index], use_cache=False).loss)
            interpolated = InterpolatedLlamaLayers(wrapper.model, layer_weights, weights, costs,
                                                   control_device=device, groups=groups).setup()
            search_started = time.monotonic()
            try:
                interpolated.retract(args.target_bits)
                probs, history = projected_gumbel_search(
                    interpolated, list(range(len(text))), args.target_bits, steps=args.steps, lr=args.lr,
                    objective=objective, gumbel_samples=args.gumbel_samples, log_every=10,
                    batch_weights=[ids.numel() for ids in text])
            finally:
                interpolated.cleanup()
                if args.objective == 'kl':
                    objective.cleanup()
            search_seconds = time.monotonic() - search_started
            logits = interpolated.alpha.detach().cpu()

    # Persist the search result before rounding: the search is the expensive
    # part, and a failure in the cheap tail must not force another run.
    if args.backend != 'mlx':
        torch.save({'probs': probs.detach().cpu(), 'logits': logits,
                    'history': history,
                    'weights': weights.cpu(), 'costs': costs.cpu(), 'names': names, 'groups': groups,
                    'target_bits': args.target_bits}, out / 'probs.pt')

    assignment = budget_constrained_argmax(logits, weights, costs, args.target_bits)
    exact = achieved_bits(assignment, weights, costs)

    projection_bits = {name: options[option] for group, option in zip(groups, assignment.tolist())
                       for name in group}
    layer_bits = ({block.index: projection_bits[block.names[0]] for block in blocks}
                  if args.grouping == 'block' else None)

    result = {
        'model': args.model,
        'target_bits': args.target_bits,
        'options': list(options),
        'projection_bits': projection_bits,
        'group_size': args.group_size,
        'preset': args.preset,
        'device': str(device),
        'backend': args.backend,
        'seed': args.seed,
        'teacher_cache': args.teacher_cache,
        'calibration_tokens': sum(document['input_ids'].__len__() for document in calib_docs),
        'objective': args.objective,
        'grouping': args.grouping,
        'gumbel_samples': args.gumbel_samples,
        'calibration_split': 'validation',
        'calibration_sequences': len(calib_docs),
        'dp_resolution': 500,
        'budget_unit': 'quantized_projection_code_bits',
        'projection_count': len(names),
        'group_count': len(groups),
        'steps': args.steps,
        'drift_max': max(record['drift'] for record in history),
        'drift_final': history[-1]['drift'],
        'projection_bits_per_parameter': exact,
        'width_counts': {f'W{width}': sum(options[option] == width for option in assignment.tolist())
                         for width in options},
        'search_seconds': search_seconds,
        'history': history,
    }
    if layer_bits is not None:
        result['layer_bits'] = {str(index): width for index, width in layer_bits.items()}
        result['block_bits_per_parameter'] = exact
    (out / 'assignment.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key != 'history'}, indent=2))



if __name__ == '__main__':
    main()
