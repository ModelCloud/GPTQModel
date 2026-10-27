# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0

"""Exercise rank-local capture, a single checkpoint writer and packed resume."""

from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _model_worker(rank, rendezvous, directory, reuse_capture):
    from transformers import LlamaConfig, LlamaForCausalLM

    from gptqmodel.looper import gsq_training_model as staged
    from gptqmodel.quantization import GSQTrainingConfig
    from gptqmodel.quantization.gsq_distributed import GSQDistributed

    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2)
    distributed = GSQDistributed(device='cpu')
    try:
        config = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128,
                             num_attention_heads=4, num_key_value_heads=2, num_hidden_layers=2)
        config._attn_implementation = 'sdpa'

        def model():
            torch.manual_seed(7)
            return LlamaForCausalLM(config).float().eval()

        documents = [{'input_ids': list(range(i + 1, i + 9))} for i in range(8)]
        validation = [{'input_ids': list(range(i + 32, i + 40))} for i in range(4)]
        initialization = [{'input_ids': list(range(i + 64, i + 72))} for i in range(2)]
        capture_counts = []
        original_capture = staged.capture_llama_gsq_inputs

        def capture(model, documents, **kwargs):
            capture_counts.append(len(documents))
            return original_capture(model, documents, **kwargs)

        staged.capture_llama_gsq_inputs = capture
        kwargs = {
            'bits': 3, 'group_size': 32,
            'gsq': GSQTrainingConfig(enabled=True, epochs=2, qk_steps=2, batch_size=4, microbatch_size=1),
            'initialization_documents': initialization, 'validation_documents': validation,
            'checkpoint_directory': Path(directory)/'ckpt', 'distributed': distributed,
            'target_cache': 'cpu', 'activation_checkpointing': True,
            'reuse_capture': reuse_capture,
        }
        first = model()
        result = staged.quantize_llama_gsq_model(first, documents, **kwargs)
        assert result['state'] == 'complete'
        assert capture_counts == (([4, 2, 2] if rank == 0 else [4, 2]) * (1 if reuse_capture else 2))
        for block in result['blocks']:
            for stage in ('attention', 'mlp'):
                assert len(block['stages'][stage]['history']) == 4  # 8/global batch4 * epochs2
                assert block['stages'][stage]['restored_best_validation_checkpoint']
        # Compare complete packed payloads across ranks, not just a loss scalar.
        for tensor in first.state_dict().values():
            expected = tensor.clone()
            dist.broadcast(expected, src=0)
            torch.testing.assert_close(tensor, expected, rtol=0, atol=0)

        def no_capture(*args, **kwargs):
            raise AssertionError('a resumed packed block must not capture or retrain')

        staged.capture_llama_gsq_inputs = no_capture
        resumed = model()
        second = staged.quantize_llama_gsq_model(resumed, documents, **kwargs)
        assert second['state'] == 'complete'
        for name, tensor in first.state_dict().items():
            torch.testing.assert_close(resumed.state_dict()[name], tensor, rtol=0, atol=0)
        # Explicit prefix migration preserves the packed bytes and provenance.
        kwargs['resume_prefix_directory'] = kwargs['checkpoint_directory']
        kwargs['checkpoint_directory'] = Path(directory)/'imported'
        imported = model()
        staged.quantize_llama_gsq_model(imported, documents, **kwargs)
        for name, tensor in first.state_dict().items():
            torch.testing.assert_close(imported.state_dict()[name], tensor, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason='requires Gloo')
@pytest.mark.parametrize('reuse_capture', [False, True])
def test_distributed_model_capture_checkpoint_and_resume(tmp_path, reuse_capture):
    mp.spawn(_model_worker, args=(f'file://{tmp_path}/rendezvous', str(tmp_path), reuse_capture), nprocs=2, join=True)
    assert sorted(p.name for p in (tmp_path/'ckpt').iterdir()) == ['layer-00.pt', 'layer-01.pt']
    for checkpoint in (tmp_path/'imported').glob('*.pt'):
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        assert saved['signature']['distributed_world_size'] == 2
        assert saved['record']['imported_from'] == str(tmp_path/'ckpt'/checkpoint.name)


@pytest.mark.parametrize('offload', [False, True])
def test_incremental_capture_matches_full_packed_prefix(offload):
    """The oracle replays the model itself after each quantized installation."""
    import copy

    from transformers import LlamaConfig, LlamaForCausalLM

    from gptqmodel.looper.gsq_training_model import quantize_llama_gsq_model
    from gptqmodel.quantization import GSQTrainingConfig

    torch.manual_seed(29)
    config = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128,
                         num_attention_heads=4, num_key_value_heads=2, num_hidden_layers=3)
    config._attn_implementation = 'sdpa'
    model = LlamaForCausalLM(config).float().eval()
    oracle = copy.deepcopy(model)
    docs = [{'input_ids': list(range(start, start+8))} for start in range(1, 5)]
    kwargs = {'bits': 3, 'group_size': 32, 'offload_capture': offload,
                  'gsq': GSQTrainingConfig(enabled=True, epochs=2, qk_steps=2, batch_size=2, microbatch_size=1),
                  'initialization_documents': docs[:2], 'validation_documents': docs[2:]}
    reference = quantize_llama_gsq_model(oracle, docs, **kwargs)
    actual = quantize_llama_gsq_model(model, docs, reuse_capture=True, **kwargs)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, oracle.state_dict()[name], rtol=0, atol=0)
    for block, expected in zip(actual['blocks'], reference['blocks']):
        for stage in ('attention', 'mlp'):
            assert block['stages'][stage]['hard_loss_after'] == expected['stages'][stage]['hard_loss_after']
