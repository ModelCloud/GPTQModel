import pytest
import torch

from gptqmodel.quantization.gsq_batching import collate_llama_documents, llama_stage_batches


@pytest.mark.parametrize('explicit_mask', [False, True])
def test_variable_length_llama_microbatch_preserves_valid_outputs(explicit_mask):
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding

    config = LlamaConfig(hidden_size=32, intermediate_size=64, num_attention_heads=4,
                         num_key_value_heads=2, num_hidden_layers=1)
    config._attn_implementation = 'eager'
    layer = LlamaDecoderLayer(config, 0).eval()
    rope = LlamaRotaryEmbedding(config)
    documents = []
    for length in (3, 7, 2):
        hidden = torch.randn(1, length, 32)
        positions = torch.arange(length).unsqueeze(0)
        mask = torch.full((length, length), torch.finfo(hidden.dtype).min).triu(1)[None, None]
        documents.append((hidden, dict(position_ids=positions, position_embeddings=rope(hidden, positions),
                                      attention_mask=mask if explicit_mask else None, use_cache=False)))
    hidden, kwargs, valid = collate_llama_documents(documents)
    with torch.no_grad():
        actual = layer(hidden, **kwargs)
        for index, (inputs, metadata) in enumerate(documents):
            expected = layer(inputs, **metadata)
            torch.testing.assert_close(actual[index, valid[index]], expected[0], rtol=1e-5, atol=1e-6)
    from gptqmodel.quantization.gsq_training import reconstruction_stage_loss

    weight = (layer.self_attn.v_proj.weight.detach()+.01).requires_grad_()
    replacements = {'self_attn.v_proj.weight': weight}
    loss = reconstruction_stage_loss(layer, (hidden,), kwargs, student_weights=replacements, output_mask=valid)
    total = sum(inputs.numel() for inputs, _ in documents)
    reference = sum(reconstruction_stage_loss(layer, (inputs,), metadata, student_weights=replacements)
                    * inputs.numel()/total for inputs, metadata in documents)
    torch.testing.assert_close(loss, reference, rtol=1e-4, atol=1e-8)
    gradient, = torch.autograd.grad(loss, weight)
    expected_gradient, = torch.autograd.grad(reference, weight)
    torch.testing.assert_close(gradient, expected_gradient, rtol=1e-4, atol=1e-7)
    batches = llama_stage_batches(documents, batch_size=3, microbatch_size=2)
    assert len(batches) == 1
    assert [batch[0][0].shape[0] for batch in batches[0]] == [2, 1]
    assert [batch[1] for batch in batches[0]] == [10*32, 2*32]
    paper_batches = llama_stage_batches([documents[0]]*512, batch_size=64, microbatch_size=16)
    assert len(paper_batches) == 8
    assert all(len(batch) == 4 for batch in paper_batches)
    assert all(microbatch[0][0].shape[0] == 16 for batch in paper_batches for microbatch in batch)
    with pytest.raises(ValueError, match='exceeds'):
        llama_stage_batches(documents, batch_size=1, microbatch_size=2)


def test_batched_staged_fit_packs_variable_length_documents():
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding
    from gptqmodel.quantization import GSQTrainingConfig
    from gptqmodel.quantization.gsq_training import quantize_llama_gsq_block

    torch.manual_seed(7)
    config = LlamaConfig(hidden_size=32, intermediate_size=64, num_attention_heads=4,
                         num_key_value_heads=2, num_hidden_layers=1)
    config._attn_implementation = 'eager'
    layer = LlamaDecoderLayer(config, 0).eval()
    rope = LlamaRotaryEmbedding(config)
    documents = []
    for length in (3, 7, 2, 5, 4):
        hidden = torch.randn(1, length, 32)
        positions = torch.arange(length).unsqueeze(0)
        documents.append((hidden, dict(position_ids=positions, position_embeddings=rope(hidden, positions),
                                      attention_mask=None, use_cache=False)))
    settings = GSQTrainingConfig(enabled=True, epochs=2, qk_steps=2, batch_size=4, microbatch_size=2)
    packed, record = quantize_llama_gsq_block(layer, documents, bits=2, group_size=32, gsq=settings)
    for stage in ('attention', 'mlp'):
        # Two optimizer batches per epoch, including the final single document.
        assert len(record['stages'][stage]['history']) == 4
    assert record['gsq_training']['batch_size'] == 4
    assert record['gsq_training']['microbatch_size'] == 2
    for hidden, kwargs in documents:
        assert torch.isfinite(packed(hidden, **kwargs)).all()
