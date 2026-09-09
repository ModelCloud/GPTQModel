"""Right-pad captured Llama documents for staged reconstruction updates."""

import torch


def collate_llama_documents(documents):
    """Preserve eager-attention masks/rotary values and return a loss mask."""
    if not documents:
        raise ValueError('GSQ microbatch must contain documents')
    first = documents[0][0]
    if first.ndim != 3 or first.shape[0] != 1 or not first.is_floating_point():
        raise ValueError('GSQ batching requires single-document floating hidden states')
    width = first.shape[-1]
    lengths = [hidden.shape[1] for hidden, _ in documents]
    maximum = max(lengths)
    hidden_batch = first.new_zeros(len(documents), maximum, width)
    valid = torch.zeros(len(documents), maximum, device=first.device, dtype=torch.bool)
    mask = first.new_full((len(documents), 1, maximum, maximum), torch.finfo(first.dtype).min)
    positions = torch.zeros(len(documents), maximum, device=first.device, dtype=torch.long)
    rotary = None
    for index, ((hidden, kwargs), length) in enumerate(zip(documents, lengths)):
        if (hidden.shape != (1, length, width) or length < 1 or hidden.dtype != first.dtype
                or hidden.device != first.device):
            raise ValueError('GSQ document geometry, dtype or device mismatch')
        unknown = set(kwargs)-{'attention_mask', 'position_ids', 'position_embeddings', 'use_cache',
                               'past_key_values'}
        if unknown or kwargs.get('use_cache') or kwargs.get('past_key_values') is not None:
            raise ValueError('Unsupported captured Llama metadata for GSQ batching')
        hidden_batch[index, :length] = hidden[0]
        valid[index, :length] = True
        captured_mask = kwargs.get('attention_mask')
        if captured_mask is None:
            # Eager Llama applies no causal restriction without an explicit
            # mask. Preserve the actual decoder call, not an assumed policy.
            captured_mask = first.new_zeros(length, length)
        elif captured_mask.shape == (1, 1, length, length) and captured_mask.is_floating_point():
            captured_mask = captured_mask[0, 0]
        else:
            raise ValueError('GSQ batching requires an additive square attention mask')
        mask[index, 0, :length, :length] = captured_mask
        # Give padded queries a finite attention destination; their outputs are excluded.
        mask[index, 0, length:, 0] = 0
        position = kwargs.get('position_ids')
        if position is None:
            position = torch.arange(length, device=first.device).unsqueeze(0)
        if position.shape != (1, length):
            raise ValueError('GSQ batching requires aligned captured positions')
        positions[index, :length] = position[0]
        embeddings = kwargs.get('position_embeddings')
        if not isinstance(embeddings, (tuple, list)) or len(embeddings) != 2:
            raise ValueError('GSQ batching requires captured rotary embeddings')
        if rotary is None:
            rotary = tuple(value.new_zeros(len(documents), maximum, value.shape[-1]) for value in embeddings)
        for destination, value in zip(rotary, embeddings):
            if value.shape != (1, length, destination.shape[-1]):
                raise ValueError('GSQ batching requires aligned rotary embeddings')
            destination[index, :length] = value[0]
    return hidden_batch, dict(attention_mask=mask, position_ids=positions,
                              position_embeddings=rotary, use_cache=False), valid


def llama_stage_batches(documents, *, batch_size, microbatch_size):
    """Group actual forward microbatches; retain and weight partial batches."""
    for value in (batch_size, microbatch_size):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError('GSQ batch sizes must be positive integers')
    if microbatch_size > batch_size:
        raise ValueError('GSQ microbatch size exceeds optimizer batch size')
    batches = []
    for start in range(0, len(documents), batch_size):
        selected = documents[start:start+batch_size]
        microbatches = []
        for offset in range(0, len(selected), microbatch_size):
            chunk = selected[offset:offset+microbatch_size]
            hidden, kwargs, mask = collate_llama_documents(chunk)
            count = sum(value.numel() for value, _ in chunk)
            microbatches.append(((hidden, kwargs, mask), count))
        batches.append(microbatches)
    return batches
