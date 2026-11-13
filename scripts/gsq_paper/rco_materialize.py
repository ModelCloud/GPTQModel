# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Materialize an RCO assignment into a loadable model for held-out scoring.

Install each selected projection from its staged width arm, then run normal
staged export. Both legacy block assignments and projection assignments are
accepted. Per-module widths are preserved in the checkpoint's dynamic config.

    python scripts/gsq_paper/rco_materialize.py \
        --model /monster/data/model/Llama-3.2-1B-Instruct \
        --assignment /monster/data/model/gsq-bench/rco/rco-2375/assignment.json \
        --arms '/monster/data/model/gsq-bench/w{bits}gsq4096/ckpt' \
        --out /monster/data/model/gsq-bench/rco-2375/model
"""

import argparse
import json
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', required=True)
    parser.add_argument('--assignment', required=True)
    parser.add_argument('--arms', required=True, help='template with {bits}')
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='auto', help='Torch device; auto prefers CUDA, then MPS, then CPU')
    return parser.parse_args()


def assignment_projection_bits(payload, blocks):
    """Validate an assignment against the independently resolved model inventory."""
    expected = {name for block in blocks for name in block.names}
    if 'projection_bits' in payload:
        result = dict(payload['projection_bits'])
    else:
        layer_bits = payload.get('layer_bits') or {}
        if set(layer_bits) != {str(block.index) for block in blocks}:
            raise ValueError('RCO materialization requires every projection of every decoder block')
        result = {name: layer_bits[str(block.index)] for block in blocks for name in block.names}
    if set(result) != expected:
        missing, extra = sorted(expected - set(result)), sorted(set(result) - expected)
        raise ValueError('RCO materialization requires every projection of every decoder block; '
                         f'missing={missing}, extra={extra}')
    if any(isinstance(width, bool) or not isinstance(width, int) or width not in (2, 3, 4)
           for width in result.values()):
        raise ValueError('RCO staged materialization requires W2/W3/W4')
    return result


def restore_assignment_block(layer, block, widths, arms, *, group_size, device,
                             projections):
    """Merge selected packed tensors, checking the shared non-projection state."""
    from gptqmodel.looper.gsq_training_model import _restore_packed_layer

    if not projections or set(widths) != set(projections):
        raise ValueError('RCO widths must cover every declared projection in the block')
    states = {}
    for width in sorted(set(widths.values())):
        checkpoint = Path(arms.format(bits=width)) / f'layer-{block:02d}.pt'
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if (saved.get('layer_index') != block or saved['signature'].get('bits') != width
                or saved['signature']['group_size'] != group_size):
            raise ValueError(f'Block, width or group size mismatch in {checkpoint}')
        states[width] = saved['state_dict']

    def shared(key):
        return not any(key.startswith(name + '.') for name in projections)

    first = states[min(states)]
    merged = {key: value for key, value in first.items() if shared(key)}
    for state in states.values():
        other = {key: value for key, value in state.items() if shared(key)}
        if other.keys() != merged.keys() or any(not torch.equal(other[key], value) for key, value in merged.items()):
            raise ValueError('RCO width arms disagree on shared block state')
    for name, width in widths.items():
        merged.update({key: value for key, value in states[width].items() if key.startswith(name + '.')})
    return _restore_packed_layer(layer, merged, names=tuple(projections), bits=widths,
                                 group_size=group_size, device=device)


def main():
    args = parse_args()
    out = Path(args.out)
    payload = json.loads(Path(args.assignment).read_text())
    print(f"target={payload['target_bits']} widths={payload['width_counts']}", flush=True)

    from gptqmodel import GPTQModel
    from gptqmodel.looper.gsq_training_model import (
        finalize_llama_gsq_wrapper,
    )
    from gptqmodel.quantization import FORMAT, GPTQConfig
    from gptqmodel.quantization.rco import discover_blocks
    from gptqmodel.utils.torch import resolve_torch_device, torch_empty_cache

    device = resolve_torch_device(args.device)
    base = min(payload.get('projection_bits', payload.get('layer_bits', {})).values())
    group_size = payload.get('group_size', 128)
    config = GPTQConfig(bits=base, group_size=group_size, sym=True, desc_act=False,
                        act_group_aware=False, format=FORMAT.GPTQ_V2, device=device,
                        offload_to_disk=False)
    wrapper = GPTQModel.load(args.model, quantize_config=config, dtype=torch.bfloat16,
                             attn_implementation='sdpa')
    model = wrapper.model.to(device)

    blocks = discover_blocks(wrapper)
    projection_bits = assignment_projection_bits(payload, blocks)
    for block in blocks:
        widths = {name: projection_bits[full_name] for name, full_name in zip(block.projections, block.names)}
        restored = restore_assignment_block(
            model.get_submodule(block.path), block.index, widths, args.arms, group_size=group_size,
            device=device, projections=block.projections)
        model.set_submodule(block.path, restored)
        print(f'  block {block.index:02d} <- {widths}', flush=True)

    run = {
        'state': 'complete',
        'layer_indices': [block.index for block in blocks],
        'bits': base,
        'projection_bits': projection_bits,
        'group_size': group_size,
        'gsq_training': payload.get('recipe') or {'enabled': True},
    }
    model.gsq_training_runs = [run]
    finalize_llama_gsq_wrapper(wrapper, run)
    out.mkdir(parents=True, exist_ok=True)
    wrapper.save_quantized(str(out))
    print(f'saved {out}', flush=True)

    # Pin placement explicitly while verifying that dynamic widths reload.
    reloaded = GPTQModel.load(str(out), backend=None, device=device, dtype=torch.float16,
                              attn_implementation='sdpa')
    reloaded.model.to(device).eval()
    mismatched = []
    for name, width in sorted(projection_bits.items()):
        module = reloaded.model.get_submodule(name)
        if int(module.bits) != width:
            mismatched.append((name, int(module.bits), width))
    print('reload width mismatches:', len(mismatched), flush=True)
    if mismatched:
        raise SystemExit(f'Reload did not preserve widths: {mismatched[:5]}')

    with torch.inference_mode():
        ids = torch.tensor([[1, 2, 3, 4]], device=device)
        loss = float(reloaded.model(input_ids=ids, labels=ids).loss)
    if not torch.isfinite(torch.tensor(loss)):
        raise RuntimeError('RCO materialized model produced a nonfinite reload loss')
    print(f'reloaded forward loss={loss:.4f}', flush=True)
    torch_empty_cache(device, gc=False)


if __name__ == '__main__':
    main()
