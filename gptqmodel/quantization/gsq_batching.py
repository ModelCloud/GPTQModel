# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-FileCopyrightText: 2026 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Right-pad captured Llama documents for staged reconstruction updates."""

import torch


def collate_llama_documents(documents, *, device=None, implicit_causal=False):
    """Preserve eager-attention masks/rotary values and return a loss mask."""
    if not documents:
        raise ValueError('GSQ microbatch must contain documents')
    first = documents[0][0]
    if first.ndim != 3 or first.shape[0] != 1 or not first.is_floating_point():
        raise ValueError('GSQ batching requires single-document floating hidden states')
    if not isinstance(implicit_causal, bool):
        raise TypeError('implicit_causal must be boolean')
    device = first.device if device is None else torch.device(device)
    width = first.shape[-1]
    lengths = [hidden.shape[1] for hidden, _ in documents]
    maximum = max(lengths)
    if implicit_causal and any(length != maximum for length in lengths):
        raise ValueError('Implicit-causal GSQ batching requires equal-length documents')
    hidden_batch = torch.zeros(len(documents), maximum, width, device=device, dtype=first.dtype)
    valid = torch.zeros(len(documents), maximum, device=device, dtype=torch.bool)
    mask = None if implicit_causal else torch.full(
        (len(documents), 1, maximum, maximum),
        torch.finfo(first.dtype).min,
        device=device,
        dtype=first.dtype,
    )
    positions = torch.zeros(len(documents), maximum, device=device, dtype=torch.long)
    rotary = None
    for index, ((hidden, kwargs), length) in enumerate(zip(documents, lengths)):
        if (hidden.shape != (1, length, width) or length < 1 or hidden.dtype != first.dtype
                or hidden.device != first.device):
            raise ValueError('GSQ document geometry, dtype or device mismatch')
        unknown = set(kwargs)-{'attention_mask', 'position_ids', 'position_embeddings', 'use_cache',
                               'past_key_values'}
        if unknown or kwargs.get('use_cache') or kwargs.get('past_key_values') is not None:
            raise ValueError('Unsupported captured Llama metadata for GSQ batching')
        hidden_batch[index, :length].copy_(hidden[0], non_blocking=True)
        valid[index, :length] = True
        captured_mask = kwargs.get('attention_mask')
        if implicit_causal:
            if captured_mask is not None:
                raise ValueError('Implicit-causal GSQ batching requires no explicit attention mask')
        elif captured_mask is None:
            # Eager Llama applies no causal restriction without an explicit
            # mask. Preserve the actual decoder call, not an assumed policy.
            captured_mask = torch.zeros(length, length, device=device, dtype=first.dtype)
        elif captured_mask.shape == (1, 1, length, length) and captured_mask.is_floating_point():
            captured_mask = captured_mask[0, 0].to(device)
        else:
            raise ValueError('GSQ batching requires an additive square attention mask')
        if mask is not None:
            mask[index, 0, :length, :length] = captured_mask
            # Give padded queries a finite attention destination; their outputs are excluded.
            mask[index, 0, length:, 0] = 0
        position = kwargs.get('position_ids')
        if position is None:
            position = torch.arange(length, device=device).unsqueeze(0)
        if position.shape != (1, length):
            raise ValueError('GSQ batching requires aligned captured positions')
        positions[index, :length].copy_(position[0].to(device), non_blocking=True)
        embeddings = kwargs.get('position_embeddings')
        if not isinstance(embeddings, (tuple, list)) or len(embeddings) != 2:
            raise ValueError('GSQ batching requires captured rotary embeddings')
        if rotary is None:
            rotary = tuple(torch.zeros(
                len(documents), maximum, value.shape[-1], device=device, dtype=value.dtype,
            ) for value in embeddings)
        for destination, value in zip(rotary, embeddings):
            if value.shape != (1, length, destination.shape[-1]):
                raise ValueError('GSQ batching requires aligned rotary embeddings')
            destination[index, :length].copy_(value[0].to(device), non_blocking=True)
    # A full fixed-length batch needs no output selection. Keeping the dense
    # boolean mask makes autograd scatter the complete stage output back into
    # an equally sized zero tensor, which is particularly expensive at the
    # Llama MLP width. ``None`` already means "use every output" to the staged
    # reconstruction objectives; retain the mask only when padding exists.
    output_mask = None if all(length == maximum for length in lengths) else valid
    return hidden_batch, dict(attention_mask=mask, position_ids=positions,
                              position_embeddings=rotary, use_cache=False), output_mask


class LazyLlamaStageBatches:
    """Materialize one optimizer batch on its destination device at a time."""

    def __init__(self, documents, *, batch_size, microbatch_size, device, implicit_causal):
        self.documents = documents
        self.batch_size = batch_size
        self.microbatch_size = microbatch_size
        self.device = torch.device(device)
        self.implicit_causal = implicit_causal

    def __len__(self):
        return (len(self.documents)+self.batch_size-1)//self.batch_size

    def __getitem__(self, index):
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(self):
            raise IndexError(index)
        start = index*self.batch_size
        selected = self.documents[start:start+self.batch_size]
        microbatches = []
        for offset in range(0, len(selected), self.microbatch_size):
            chunk = selected[offset:offset+self.microbatch_size]
            hidden, kwargs, mask = collate_llama_documents(
                chunk,
                device=self.device,
                implicit_causal=self.implicit_causal,
            )
            count = sum(value.numel() for value, _ in chunk)
            microbatches.append(((hidden, kwargs, mask), count))
            del chunk
        del selected
        return microbatches


def llama_stage_batches(documents, *, batch_size, microbatch_size, device=None, implicit_causal=False, lazy=False):
    """Group actual forward microbatches; retain and weight partial batches."""
    for value in (batch_size, microbatch_size):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError('GSQ batch sizes must be positive integers')
    if microbatch_size > batch_size:
        raise ValueError('GSQ microbatch size exceeds optimizer batch size')
    if lazy:
        if device is None:
            raise ValueError('Lazy GSQ batching requires a destination device')
        return LazyLlamaStageBatches(
            documents,
            batch_size=batch_size,
            microbatch_size=microbatch_size,
            device=device,
            implicit_causal=implicit_causal,
        )
    batches = []
    for start in range(0, len(documents), batch_size):
        selected = documents[start:start+batch_size]
        microbatches = []
        for offset in range(0, len(selected), microbatch_size):
            chunk = selected[offset:offset+microbatch_size]
            hidden, kwargs, mask = collate_llama_documents(chunk, device=device, implicit_causal=implicit_causal)
            count = sum(value.numel() for value, _ in chunk)
            microbatches.append(((hidden, kwargs, mask), count))
        batches.append(microbatches)
    return batches
