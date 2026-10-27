# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Materialize an RCO assignment into a loadable model for held-out scoring.

Staged GSQ trains each decoder block independently and writes one checkpoint
per block, so a non-uniform assignment needs no retraining: install block
``n`` from the arm whose width the assignment chose for ``n``, then run the
normal staged export. The result is an ordinary mixed-width GPTQModel
checkpoint whose per-module widths live in ``dynamic``.

    python scripts/gsq_paper/rco_materialize.py \
        --model /monster/data/model/Llama-3.2-1B-Instruct \
        --assignment /monster/data/model/gsq-bench/rco/rco-2375/assignment.json \
        --arms '/monster/data/model/gsq-bench/w{bits}gsq4096/ckpt' \
        --out /monster/data/model/gsq-bench/rco-2375/model
"""

import argparse
import contextlib
import json
import os
from pathlib import Path

import torch

PROJECTIONS = ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
               'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj')


@contextlib.contextmanager
def _one_visible_device(device):
    """Restrict visible accelerators so load-time placement cannot split."""
    original = os.environ.get('ROCR_VISIBLE_DEVICES'), os.environ.get('HIP_VISIBLE_DEVICES')
    index = device.split(':')[-1] if ':' in device else '0'
    os.environ['ROCR_VISIBLE_DEVICES'] = index
    os.environ['HIP_VISIBLE_DEVICES'] = index
    try:
        yield
    finally:
        for name, value in zip(('ROCR_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES'), original):
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', required=True)
    parser.add_argument('--assignment', required=True)
    parser.add_argument('--arms', required=True, help='template with {bits}')
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    payload = json.loads(Path(args.assignment).read_text())
    layer_bits = {int(index): int(width) for index, width in payload['layer_bits'].items()}
    print(f"target={payload['target_bits']} widths={payload['width_counts']}", flush=True)

    from gptqmodel import GPTQModel
    from gptqmodel.looper.gsq_training_model import (
        _restore_packed_layer,
        finalize_llama_gsq_wrapper,
    )
    from gptqmodel.models._const import DEVICE
    from gptqmodel.quantization import FORMAT, GPTQConfig

    base = min(layer_bits.values())
    config = GPTQConfig(bits=base, group_size=128, sym=True, desc_act=False,
                        act_group_aware=False, format=FORMAT.GPTQ_V2, device=DEVICE.CUDA,
                        offload_to_disk=False)
    wrapper = GPTQModel.load(args.model, quantize_config=config, dtype=torch.bfloat16,
                             attn_implementation='sdpa')
    model = wrapper.model.to(args.device)

    for block, width in sorted(layer_bits.items()):
        checkpoint = Path(args.arms.format(bits=width)) / f'layer-{block:02d}.pt'
        if not checkpoint.exists():
            raise SystemExit(f'Missing block checkpoint: {checkpoint}')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if saved['signature'].get('bits') != width:
            raise SystemExit(f'{checkpoint} is not W{width}')
        restored = _restore_packed_layer(
            model.model.layers[block], saved['state_dict'], names=PROJECTIONS, bits=width,
            group_size=saved['signature']['group_size'], device=args.device)
        model.model.layers[block] = restored
        print(f'  block {block:02d} <- W{width}', flush=True)

    run = {
        'state': 'complete',
        'layer_indices': list(range(len(layer_bits))),
        'bits': {block: width for block, width in layer_bits.items()},
        'layer_bits': dict(layer_bits),
        'group_size': 128,
        'gsq_training': payload.get('recipe') or {'enabled': True},
    }
    model.gsq_training_runs = [run]
    finalize_llama_gsq_wrapper(wrapper, run)
    out.mkdir(parents=True, exist_ok=True)
    wrapper.save_quantized(str(out))
    print(f'saved {out}', flush=True)

    # Verify the widths survive a reload: load-time resolution reads `dynamic`.
    # A single visible device keeps auto-placement from splitting this small
    # model across cards, which would break the first normalization.
    with _one_visible_device(args.device):
        reloaded = GPTQModel.load(str(out), backend=None, dtype=torch.bfloat16,
                                  attn_implementation='sdpa')
        reloaded.model.to(args.device).eval()
    mismatched = []
    for block, width in sorted(layer_bits.items()):
        for name in PROJECTIONS:
            module = reloaded.model.get_submodule(f'model.layers.{block}.{name}')
            if int(module.bits) != width:
                mismatched.append((block, name, int(module.bits), width))
    print('reload width mismatches:', len(mismatched), flush=True)
    if mismatched:
        raise SystemExit(f'Reload did not preserve widths: {mismatched[:5]}')

    with torch.inference_mode():
        ids = torch.tensor([[1, 2, 3, 4]], device=args.device)
        loss = float(reloaded.model(input_ids=ids, labels=ids).loss)
    print(f'reloaded forward loss={loss:.4f} finite={bool(torch.isfinite(torch.tensor(loss)))}', flush=True)


if __name__ == '__main__':
    main()
