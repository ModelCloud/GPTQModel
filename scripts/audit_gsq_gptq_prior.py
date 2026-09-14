"""Compare signed repository GPTQ with the pinned author on real NM512 inputs."""

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

from scripts.validate_qvq_gsq_layers import digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--author', type=Path, required=True)
    parser.add_argument('--lion-wheel', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    inventory = 'CPU'
    if args.device == 'cuda':
        uuid = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if not uuid.startswith('GPU-') or ',' in uuid or not os.environ.get('GPU_ALLOCATOR_LEASE_ID'):
            raise ValueError('CUDA audit requires an exclusive UUID lease')
        idle = 0
        for _ in range(60):
            inventory = subprocess.check_output(['nvidia-smi', '--id='+uuid,
                '--query-gpu=uuid,name,pci.bus_id,memory.used,utilization.gpu',
                '--format=csv,noheader,nounits'], text=True).strip()
            fields = [v.strip() for v in inventory.split(',')]
            processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                                 '--format=csv,noheader'], text=True)
            idle = idle+1 if int(fields[-2]) <= 8 and int(fields[-1]) == 0 and uuid not in processes else 0
            print('PREFLIGHT', inventory, idle, flush=True)
            if idle == 3:
                break
            time.sleep(1)
        else:
            raise ValueError('Idle preflight timeout')
    sources = [Path(__file__), Path('gptqmodel/quantization/gptq.py'),
               Path('gptqmodel/quantization/config.py'), Path('gptqmodel/quantization/quantizer.py'),
               Path('gptqmodel/quantization/gsq_initialization.py'),
               args.author/'src/prior/gptq.py', args.author/'src/prior/quant.py']
    source_hashes = {str(p.resolve()): digest(p) for p in sources}
    (args.output/'executed-source').mkdir()
    for i, source in enumerate(sources):
        shutil.copy2(source, args.output/'executed-source'/f'{i}_{source.name}')
    sys.path.insert(0, str(args.author))
    sys.path.insert(0, str(args.lion_wheel))
    import torch
    from safetensors import safe_open
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaRMSNorm
    from src.prior.gptq import GPTQ as AuthorGPTQ
    from src.prior.quant import Quantizer as AuthorQuantizer
    from gptqmodel.quantization.config import GPTQConfig
    from gptqmodel.quantization.gptq import GPTQ
    from gptqmodel.quantization.gsq_initialization import SignedGSQQuantizer

    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = LlamaConfig.from_pretrained(args.model, local_files_only=True)
    with safe_open(args.model/'model.safetensors', framework='pt', device='cpu') as handle:
        embedding = handle.get_tensor('model.embed_tokens.weight').float().to(args.device)
        norm_weight = handle.get_tensor('model.layers.0.input_layernorm.weight').float()
        weight = handle.get_slice('model.layers.0.self_attn.q_proj.weight')[:32, :256].float().to(args.device)
    norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps).to(args.device).eval()
    norm.weight.data.copy_(norm_weight)
    documents = json.loads(args.inputs.read_text())['train']
    if len(documents) != 512:
        raise ValueError('Requires the locked NM512 calibration inputs')
    with torch.no_grad():
        batches = [norm(torch.nn.functional.embedding(torch.tensor([row['input_ids']], device=args.device), embedding))
                   [..., :256].contiguous() for row in documents]
    del embedding
    rows = []
    for bits in (2, 3, 4):
        layer = torch.nn.Linear(256, 32, bias=False).to(args.device)
        layer.weight.data.copy_(weight)
        author = AuthorGPTQ(copy.deepcopy(layer), 'audit_linear',
                            SimpleNamespace(quantization=SimpleNamespace(gsq_bits=bits)), args.device, torch.float32)
        author.quantizer = AuthorQuantizer()
        author.quantizer.configure(bits, perchannel=True, sym=True, mse=True)
        qcfg = GPTQConfig(bits=bits, group_size=128, sym=True, desc_act=False, act_group_aware=False,
                          damp_percent=.01, mse=2.4, scale_search='mse')
        repository = GPTQ(copy.deepcopy(layer), qcfg)
        repository.quantizer = SignedGSQQuantizer(qcfg)
        repository.quantizer.configure(perchannel=True)
        with torch.no_grad():
            for batch in batches:
                output = layer(batch)
                author.add_batch(batch, output)
                repository.add_batch(batch, output)
        expected, expected_scales = author.fasterquant(None, percdamp=.01, groupsize=128)
        actual, scales, zeros, groups, *_ = repository.quantize()
        codes_expected = torch.round(expected/expected_scales.repeat_interleave(128, dim=1))
        codes_actual = torch.round(actual/scales.repeat_interleave(128, dim=1))
        row = dict(bits=bits, weight_max_abs=(actual-expected).abs().max().item(),
                   scale_max_abs=(scales-expected_scales).abs().max().item(),
                   assignment_mismatches=int((codes_actual != codes_expected).sum()),
                   author_negative_scales=int((expected_scales < 0).sum()),
                   repository_negative_scales=int((scales < 0).sum()),
                   zero_points_correct=bool((zeros == 2**(bits-1)).all()),
                   groups_correct=torch.equal(groups.cpu(), torch.arange(256, dtype=groups.dtype)//128))
        rows.append(row)
        repository.free()
        print('PRIOR', json.dumps(row), flush=True)
    report = dict(scope='32x256 real Q-projection slice; 512 actual embedding/RMSNorm activation sequences',
                  device=args.device, inventory=inventory, source_hashes=source_hashes,
                  calibration_samples=len(batches), calibration_tokens=sum(v.shape[1] for v in batches),
                  inputs_sha256=digest(args.inputs), model_sha256=digest(args.model/'model.safetensors'),
                  qvq_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  author_commit=subprocess.check_output(['git', '-C', str(args.author),
                                                          'rev-parse', 'HEAD'], text=True).strip(),
                  driver_sha256=digest(__file__), torch=str(torch.__version__), rows=rows)
    write_json(args.output/'report.json', report)
    (args.output/'audit.md').write_text('# Signed GPTQ prior audit\n\n'
        'This is an algorithm comparison on a real projection slice, not complete-model quality evidence.\n\n'
        '```json\n'+json.dumps(report, indent=2)+'\n```\n')


if __name__ == '__main__':
    main()
