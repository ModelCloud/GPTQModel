"""Audit author scalar initialization against repository MSE range search."""

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from scripts.validate_qvq_gsq_layers import digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--author', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--adapter', choices=('positive_mse', 'signed_paper'), default='positive_mse')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    from safetensors import safe_open
    from gptqmodel.quantization.config import GPTQConfig
    from gptqmodel.quantization.quantizer import Quantizer, quantize
    from gptqmodel.quantization.gsq_initialization import signed_scalar_range_search

    torch.set_num_threads(4)
    source = args.author/'src/prior/quant.py'
    spec = importlib.util.spec_from_file_location('gsq_author_quant_reference', source)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    with safe_open(args.model/'model.safetensors', framework='pt', device='cpu') as handle:
        weight = handle.get_slice('model.layers.0.self_attn.q_proj.weight')[:256, :].float()
    rows = []
    for bits in (2, 3, 4):
        mismatch = negative = magnitude_mismatch = 0
        maximum = weight_maximum = 0.
        for start in range(0, weight.shape[1], 128):
            value = weight[:, start:start+128]
            author = reference.Quantizer()
            author.configure(bits, perchannel=True, sym=True, mse=True)
            author.find_params(value, weight=True)
            if args.adapter == 'positive_mse':
                adapter = Quantizer(GPTQConfig(bits=bits, group_size=128, sym=True, desc_act=False,
                                               act_group_aware=False, mse=2.4, scale_search='mse'))
                adapter.configure(perchannel=True)
                adapter.find_params(value, weight=True)
            else:
                _, scale, zero = signed_scalar_range_search(value, bits=bits)
                adapter = SimpleNamespace(scale=scale, zero=zero, maxq=2**bits-1)
            mismatch += int((author.scale != adapter.scale).sum())
            negative += int((author.scale < 0).sum())
            magnitude_mismatch += int((author.scale.abs() != adapter.scale.abs()).sum())
            maximum = max(maximum, (author.scale-adapter.scale).abs().max().item())
            expected = reference.quantize(value, author.scale, author.zero, author.maxq)
            actual = quantize(value, adapter.scale, adapter.zero, adapter.maxq, False)
            weight_maximum = max(weight_maximum, (expected-actual).abs().max().item())
        rows.append(dict(bits=bits, groups=weight.shape[1]//128, rows=len(weight),
                         scale_mismatches=mismatch, author_negative_scales=negative,
                         magnitude_mismatches=magnitude_mismatch, scale_max_abs=maximum,
                         quantized_weight_max_abs=weight_maximum))
    report = dict(scope='CPU initializer range-search arithmetic; no model-quality claim',
                  qvq_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  author_commit=subprocess.check_output(['git', '-C', str(args.author),
                                                          'rev-parse', 'HEAD'], text=True).strip(),
                  author_source_sha256=digest(source), model_sha256=digest(args.model/'model.safetensors'),
                  driver_sha256=digest(__file__), torch=str(torch.__version__),
                  signed_implementation_sha256=digest('gptqmodel/quantization/gsq_initialization.py'),
                  projection='model.layers.0.self_attn.q_proj.weight', shape=list(weight.shape),
                  candidate_config=dict(adapter=args.adapter, mse=2.4, group_size=128, sym=True), rows=rows)
    write_json(args.output/'report.json', report)
    (args.output/'audit.md').write_text('# Initializer scale parity\n\n'
        'Range-search arithmetic on real model weights. This does not establish full GPTQ trajectory '
        'parity or quantify model quality.\n\n```json\n'+json.dumps(report, indent=2)+'\n```\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
