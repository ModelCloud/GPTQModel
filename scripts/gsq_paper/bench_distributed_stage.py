# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Complete GSQ MLP updates, with fixed global batches and fresh microbatch noise.

Run from this checkout with its root on PYTHONPATH. Examples (reserve devices
with the other users of the machine before launching):

  python -m torch.distributed.run --standalone --nproc-per-node=2 \
    scripts/gsq_paper/bench_distributed_stage.py --bits 2 --cached-targets device
  python -m torch.distributed.run --standalone --nproc-per-node=4 \
    scripts/gsq_paper/bench_distributed_stage.py --bits 3 --microbatch-size 16

Use the same arguments at world sizes 1, 2, 4, and 8 to compare timings ONLY
when the global microbatch boundaries also match. Global batch 64 / microbatch
16 has just four independent draws and cannot be distributed over eight ranks.
The default is 64 documents, microbatch 2, sequence 4096, BF16, and the complete
4096 -> 14336 -> 4096 Llama MLP (gate, up, down, RMSNorm, residual, MSE).

Without --model, weights and activations are synthetic; this is a geometry
benchmark, not a model quality evaluation. --model reads a local safetensors
checkpoint's one decoder layer without loading/saving the rest of the model.
No model or activation artifacts are written. Cached targets include the full
teacher block and the fixed attention residual; their construction is reported
separately. --cached-targets none reruns the full teacher and student block.

Matched draws are generated lazily per (update, global microbatch, projection),
never shared between microbatches. Explicit uniforms exercise the public Torch
relaxation path. --draws native measures the production RNG/fast-path dispatch
with distinct rank seeds; that mode is NOT a matched-noise topology comparison.
Correctness, including an independent float64 oracle and Lion state, lives in
tests/test_gsq_distributed_stage.py (also a torchrun entry point).

Pinned reference audit: /monster/home/gsq-ref at
03fc16484c369e3127225615d5e03e8d3a6043e3. src/data/dataset.py shards documents
contiguously, src/trainer.py divides global batch by world size and redraws
quantizers inside each training microbatch. Its floor-division loops discard
remainders; this sidecar retains every document and weights output elements.
Its local default config has 10 epochs (640 updates for 4096 / 64), whereas
the paper-arm budget used here is 20 epochs (1280 updates). Neither is a timing
claim, and a profiler's invocation count is not an optimizer-update count.
"""

import argparse
import gc
import json
import math
import os
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from gptqmodel.quantization import gsq_training as stage

REFERENCE_REVISION = '03fc16484c369e3127225615d5e03e8d3a6043e3'


def microbatch_plan(global_batch, microbatch_size, world_size=1):
    """Partition whole microbatches, preserving exactly the single-device draws."""
    if min(global_batch, microbatch_size, world_size) < 1:
        raise ValueError('batch, microbatch, and world size must be positive')
    chunks = [(i, start, min(start + microbatch_size, global_batch))
              for i, start in enumerate(range(0, global_batch, microbatch_size))]
    if len(chunks) < world_size:
        raise ValueError('fewer global microbatches than ranks; change microbatch size explicitly')
    return [chunks[len(chunks)*rank//world_size:len(chunks)*(rank+1)//world_size]
            for rank in range(world_size)]


def draw_seed(seed, update, microbatch, projection):
    # Independent of world size, device ordinal, and Python's salted hash().
    return (seed + 1_000_003*(update+1) + 10_007*(microbatch+1) + 101*projection) % (2**63-1)


class MicrobatchUniforms:
    """The explicit-uniform API needs indexing, not a bank of all weight draws."""

    def __init__(self, quantizers, chunks, *, seed, update):
        self.quantizers, self.chunks = quantizers, chunks
        self.seed, self.update = seed, update

    def __getitem__(self, index):
        owner = self
        global_index = self.chunks[index][0]

        class ProjectionDraws:
            def __getitem__(self, name):
                quantizer = owner.quantizers[name]
                projection = list(owner.quantizers).index(name)
                generator = torch.Generator(device=quantizer.logits.device).manual_seed(
                    draw_seed(owner.seed, owner.update, global_index, projection))
                return torch.rand(quantizer.logits.shape, dtype=quantizer.logits.dtype,
                                  device=quantizer.logits.device, generator=generator)

        return ProjectionDraws()


def initialize_distributed(device_kind, backend='auto'):
    """torchrun environment owns topology. NCCL on ROCm dispatches to RCCL."""
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    if device_kind == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA/ROCm device required; use --device cpu for tiny correctness checks')
        device = torch.device('cuda', int(os.environ.get('LOCAL_RANK', '0')))
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
    if world_size > 1:
        if backend == 'auto':
            backend = 'nccl' if device.type == 'cuda' else 'gloo'
        dist.init_process_group(backend, timeout=timedelta(minutes=15))
        from gptqmodel.quantization.gsq_distributed import GSQDistributed

        context = GSQDistributed(device=device)
    else:
        context = None
    return device, rank, world_size, context


def _synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _all_records(record, world_size):
    if world_size == 1:
        return [record]
    records = [None] * world_size
    dist.all_gather_object(records, record)
    return records


def _build_layer(args, device):
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import (
        LlamaDecoderLayer,
        LlamaRotaryEmbedding,
    )

    if args.model:
        config = LlamaConfig.from_pretrained(args.model, local_files_only=True)
        if (config.hidden_size, config.intermediate_size) != (args.hidden, args.intermediate):
            raise ValueError('checkpoint geometry differs from --hidden/--intermediate')
    else:
        heads = max(1, args.hidden // 128)
        if args.hidden % heads or (args.hidden // heads) % 2:
            raise ValueError('hidden size must produce even rotary head dimensions')
        config = LlamaConfig(hidden_size=args.hidden, intermediate_size=args.intermediate,
                             num_hidden_layers=1, num_attention_heads=heads,
                             num_key_value_heads=math.gcd(heads, 8),
                             head_dim=args.hidden // heads, max_position_embeddings=args.sequence_length,
                             attention_dropout=0., rms_norm_eps=1e-5)
    config._attn_implementation = 'sdpa'
    dtype = getattr(torch, args.dtype)
    with torch.device('meta'):
        layer = LlamaDecoderLayer(config, layer_idx=args.layer_index).to(dtype=dtype)
    layer.to_empty(device=device)
    with torch.no_grad():
        if args.model:
            from safetensors import safe_open

            folder = Path(args.model)
            index = folder / 'model.safetensors.index.json'
            mapping = json.loads(index.read_text())['weight_map'] if index.exists() else None
            for name, parameter in layer.named_parameters():
                key = f'model.layers.{args.layer_index}.{name}'
                filename = mapping[key] if mapping is not None else 'model.safetensors'
                with safe_open(folder / filename, framework='pt', device='cpu') as source:
                    parameter.copy_(source.get_tensor(key))
        else:
            generator = torch.Generator(device=device).manual_seed(args.seed)
            for name, parameter in layer.named_parameters():
                if parameter.ndim == 1:
                    parameter.fill_(1 if name.endswith('layernorm.weight') else 0)
                else:
                    parameter.normal_(0, .02, generator=generator)
    layer.eval().requires_grad_(False)
    rotary = LlamaRotaryEmbedding(config).to(device)
    return layer, rotary


def _build_quantizers(layer, args, bits, device):
    quantizers = {}
    generator = torch.Generator(device=device).manual_seed(args.seed + 9)
    for name, projection in layer.mlp.named_children():
        if not isinstance(projection, torch.nn.Linear):
            continue
        weight = projection.weight.detach()
        rows, columns = weight.shape
        if columns % args.group_size:
            raise ValueError('group size must divide gate/up/down input widths')
        maximum = 2**(bits-1)-1
        scales = weight.float().reshape(rows, -1, args.group_size).abs().amax(-1)
        scales = (scales / maximum).clamp_min(1e-6)
        extended = scales.repeat_interleave(args.group_size, dim=1)
        codes = (weight.float()/extended).round().clamp(-maximum-1, maximum)
        initial = (codes*extended).to(weight.dtype)
        noise = torch.randn((4 if bits == 2 else 5, rows, columns), device=device,
                            dtype=weight.dtype, generator=generator)
        quantizers[f'mlp.{name}.weight'] = stage.GSQScalarTrainingModule(
            initial, scales, args.group_size, bits=bits, noise=noise,
            logits_dtype=getattr(torch, args.logits_dtype))
    if len(quantizers) != 3:
        raise ValueError('benchmark requires gate, up, and down projections')
    return quantizers


def _prepare_batches(layer, rotary, chunks, args, device):
    cached = args.cached_targets != 'none'
    storage = torch.device('cpu') if args.cached_targets == 'cpu' else device
    dtype = getattr(torch, args.dtype)
    batches = []
    attention = stage.LlamaGSQAttentionStage(layer)
    mlp = stage.LlamaGSQMLPStage(layer)
    with torch.no_grad():
        for global_index, start, end in chunks:
            generator = torch.Generator(device=device).manual_seed(args.seed + 11_111 + global_index)
            hidden = torch.randn(end-start, args.sequence_length, args.hidden, device=device,
                                 dtype=dtype, generator=generator)
            positions = torch.arange(args.sequence_length, device=device).unsqueeze(0)
            kwargs = {'position_embeddings': rotary(hidden, positions), 'use_cache': False}
            if cached:
                residual = attention(hidden, **kwargs)
                teacher = mlp(residual)
                payload = (residual.to(storage), teacher.to(storage))
            else:
                payload = (hidden, kwargs)
            if storage.type == 'cpu' and device.type == 'cuda':
                payload = tuple(t.pin_memory() for t in payload)
            batches.append((payload, (end-start)*args.sequence_length*args.hidden))

    def objective(batch, weights):
        if cached:
            residual, teacher = (value.to(device, non_blocking=True) for value in batch)
            return stage.reconstruction_cached_llama_mlp_loss(
                layer, (residual, teacher), weights,
                activation_checkpointing=args.activation_checkpointing)
        hidden, kwargs = batch
        return stage.reconstruction_stage_loss(
            layer, (hidden,), kwargs, student_weights=weights,
            activation_checkpointing=args.activation_checkpointing)

    return batches, objective


def benchmark(args, bits, device, rank, world_size, context):
    plan = microbatch_plan(args.global_batch, args.microbatch_size, world_size)
    chunks = plan[rank]
    started = time.perf_counter()
    layer, rotary = _build_layer(args, device)
    quantizers = _build_quantizers(layer, args, bits, device)
    if context is not None:
        context.sync_quantizers(quantizers)
    _synchronize(device)
    initialized_seconds = time.perf_counter()-started
    started = time.perf_counter()
    batches, objective = _prepare_batches(layer, rotary, chunks, args, device)
    _synchronize(device)
    prepared_seconds = time.perf_counter()-started
    groups = [group for quantizer in quantizers.values() for group in quantizer.optimizer_groups(
        assignment_lr=args.assignment_lr, scale_lr=args.scale_lr, weight_decay=1.)]
    optimizer = stage.GSQLion(groups, betas=(.9, .95))
    base_lrs = [group['lr'] for group in optimizer.param_groups]
    generator = torch.Generator(device=device).manual_seed(
        args.seed if context is None else context.seed(args.seed))
    budget = args.epochs*math.ceil(args.train_documents/args.global_batch)
    if args.warmup + args.repeats > budget:
        raise ValueError('warmup plus measured updates exceeds the declared training budget')
    calls = 0

    def update():
        nonlocal calls
        tau, multiplier = stage.sampling_schedule(
            calls, budget, temperature=(2., .05), multiplier=(100., 500.))
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            group['lr'] = stage.stage_learning_rate(calls, budget, base_lr=base_lr,
                                                    min_lr=.1, decay='cosine')
        uniforms = (MicrobatchUniforms(quantizers, chunks, seed=args.seed, update=calls)
                    if args.draws == 'matched' else None)
        result = stage.train_stage_update(
            quantizers, optimizer, batches, objective, generator=generator,
            temperature=tau, multiplier=multiplier, uniforms=uniforms, distributed=context)
        calls += 1
        return result

    for _ in range(args.warmup):
        update()
    _synchronize(device)
    if world_size > 1:
        dist.barrier()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    losses, durations = [], []
    started = time.perf_counter()
    for _ in range(args.repeats):
        begin = time.perf_counter()
        losses.append(update())
        _synchronize(device)
        durations.append((time.perf_counter()-begin)*1000)
    elapsed = time.perf_counter()-started
    allocated = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0
    reserved = torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else 0
    # State summaries are diagnostics, not a substitute for the elementwise
    # float64 Lion comparisons in the correctness suite. Outside timing/peak.
    lion = {name: {
        parameter_name: float(optimizer.state[parameter]['exp_avg'].float().norm())
        for parameter_name, parameter in quantizer.named_parameters()
    } for name, quantizer in quantizers.items()}
    record = {'rank': rank, 'milliseconds_per_update': 1000*elapsed/args.repeats,
                  'update_milliseconds': durations, 'peak_allocated_bytes': allocated,
                  'peak_reserved_bytes': reserved, 'measured_losses': torch.stack(losses).float().cpu().tolist(),
                  'optimizer_updates': calls, 'local_documents': sum(end-start for _, start, end in chunks),
                  'local_output_elements': sum(count for _, count in batches),
                  'microbatch_shapes': [[end-start, args.sequence_length, args.hidden] for _, start, end in chunks],
                  'initialization_seconds': initialized_seconds, 'target_preparation_seconds': prepared_seconds,
                  'lion_momentum_norms': lion}
    records = _all_records(record, world_size)
    return {'kind': 'gsq_distributed_stage_benchmark', 'bits': bits,
                'import_path': str(Path(stage.__file__).resolve()),
                'distributed_import_path': (str(Path(sys.modules[context.__class__.__module__].__file__).resolve())
                                         if context is not None else None),
                'torch_version': torch.__version__, 'hip_version': torch.version.hip,
                'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
                'backend': dist.get_backend() if world_size > 1 else 'single',
                'collective_implementation': 'RCCL' if torch.version.hip and world_size > 1 and
                dist.get_backend() == 'nccl' else None,
                'world_size': world_size, 'global_batch': args.global_batch,
                'global_tokens_per_update': args.global_batch*args.sequence_length,
                'global_output_elements_per_update': args.global_batch*args.sequence_length*args.hidden,
                'microbatch_size': args.microbatch_size,
                'global_microbatch_boundaries': [chunk for shard in plan for chunk in shard],
                'rank_microbatch_boundaries': plan,
                'tokens_per_global_microbatch': [(end-start)*args.sequence_length
                                             for shard in plan for _, start, end in shard],
                'projection_weight_shapes': {name: list(quantizer.logits.shape[1:])
                                          for name, quantizer in quantizers.items()},
                'sequence_length': args.sequence_length, 'hidden_size': args.hidden,
                'intermediate_size': args.intermediate, 'dtype': args.dtype, 'logits_dtype': args.logits_dtype,
                'teacher_weights': str(args.model) if args.model else 'synthetic random; no quality claim',
                'activation_source': 'synthetic normal', 'initializer': 'groupwise symmetric RTN',
                'objective': 'cached full-block target / fixed-attention MLP' if args.cached_targets != 'none'
                else 'full decoder teacher and student; train MLP only',
                'cached_targets': args.cached_targets, 'activation_checkpointing': args.activation_checkpointing,
                'draws': args.draws, 'matched_across_topologies': args.draws == 'matched',
                'relaxation_dispatch': 'explicit-uniform Torch' if args.draws == 'matched' else 'production',
                'warmup_updates': args.warmup, 'measured_updates': args.repeats, 'budget_updates': budget,
                'train_documents': args.train_documents, 'epochs': args.epochs,
                'milliseconds_per_update': max(r['milliseconds_per_update'] for r in records),
                'peak_allocated_bytes': max(r['peak_allocated_bytes'] for r in records),
                'peak_reserved_bytes': max(r['peak_reserved_bytes'] for r in records),
                'memory_scope': 'PyTorch allocator peaks; external RCCL allocations excluded',
                'reference_revision': REFERENCE_REVISION, 'ranks': records}


def parser():
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    result.add_argument('--backend', choices=('auto', 'gloo', 'nccl'), default='auto')
    result.add_argument('--bits', nargs='+', type=int, choices=(2, 3), default=[2, 3])
    result.add_argument('--global-batch', type=int, default=64)
    result.add_argument('--microbatch-size', type=int, default=2)
    result.add_argument('--sequence-length', type=int, default=4096)
    result.add_argument('--hidden', type=int, default=4096)
    result.add_argument('--intermediate', type=int, default=14336)
    result.add_argument('--group-size', type=int, default=128)
    result.add_argument('--dtype', choices=('bfloat16', 'float32'), default='bfloat16')
    result.add_argument('--logits-dtype', choices=('bfloat16', 'float32'), default='bfloat16')
    result.add_argument('--cached-targets', choices=('none', 'cpu', 'device'), default='device')
    result.add_argument('--activation-checkpointing', action='store_true')
    result.add_argument('--draws', choices=('matched', 'native'), default='matched')
    result.add_argument('--model', type=Path, help='local Llama safetensors checkpoint (read only)')
    result.add_argument('--layer-index', type=int, default=0)
    result.add_argument('--seed', type=int, default=7)
    result.add_argument('--warmup', type=int, default=1)
    result.add_argument('--repeats', type=int, default=3)
    result.add_argument('--train-documents', type=int, default=4096)
    result.add_argument('--epochs', type=int, default=20)
    result.add_argument('--assignment-lr', type=float, default=1e-4)
    result.add_argument('--scale-lr', type=float, default=5e-5)
    return result


def main():
    args = parser().parse_args()
    if min(args.repeats, args.sequence_length, args.hidden, args.intermediate,
           args.group_size, args.train_documents, args.epochs) < 1 or args.warmup < 0:
        raise SystemExit('positive sizes/repeats/epochs and nonnegative warmup required')
    torch.set_num_threads(1)
    device, rank, world_size, context = initialize_distributed(args.device, args.backend)
    try:
        for bits in args.bits:
            record = benchmark(args, bits, device, rank, world_size, context)
            if rank == 0:
                print('GSQ_BENCHMARK_JSON=' + json.dumps(record, allow_nan=False), flush=True)
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
