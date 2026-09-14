"""Streaming input Gram statistics for activation-aware GSQ processors."""

import math
import threading

import torch


class GSQInputGram:
    """Own one bounded FP32 Gram matrix; never retain caller activations.

    A batch contributes ``source_weight * X.T @ X``. An optional boolean mask
    selects valid tokens before flattening. ``take`` transfers the accumulated
    matrix and closes capture, preventing accidental replay double-counting.
    """

    def __init__(self, columns, *, device='cpu', max_bytes=1024**3):
        if isinstance(columns, bool) or not isinstance(columns, int) or columns <= 0:
            raise ValueError('GSQ Gram columns must be a positive integer')
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < columns*columns*4:
            raise ValueError('GSQ Gram exceeds its memory budget')
        self.columns = columns
        self.device = torch.device(device)
        self._lock = threading.Lock()
        self._gram = None
        self._closed = False
        self.tokens = 0
        self.weighted_tokens = 0.

    def add(self, inputs, *, source_weight=1., mask=None):
        if (isinstance(source_weight, bool) or not isinstance(source_weight, (int, float))
                or not math.isfinite(source_weight) or source_weight < 0):
            raise ValueError('GSQ source weight must be finite and nonnegative')
        if inputs.ndim < 2 or inputs.shape[-1] != self.columns or not inputs.is_floating_point():
            raise ValueError('GSQ capture requires floating [...,in] activations')
        if mask is not None and (mask.dtype != torch.bool or mask.shape != inputs.shape[:-1]
                                 or mask.device != inputs.device):
            raise ValueError('GSQ token mask must be boolean and match input leading dimensions/device')
        with self._lock, torch.inference_mode(False), torch.no_grad():
            if self._closed:
                raise RuntimeError('GSQ Gram capture is closed')
            # Validate only selected tokens: padding can legitimately be ignored.
            values = inputs.detach().reshape(-1, self.columns)
            if mask is not None:
                values = values[mask.reshape(-1)]
            values = values.to(device=self.device, dtype=torch.float32)
            if not torch.isfinite(values).all():
                raise ValueError('GSQ capture requires finite selected activations in FP32')
            if source_weight == 0 or values.shape[0] == 0:
                return
            contribution = (values.T @ values) * source_weight
            if not torch.isfinite(contribution).all():
                raise ValueError('GSQ calibration Gram overflow')
            updated = contribution if self._gram is None else self._gram + contribution
            if not torch.isfinite(updated).all():
                raise ValueError('GSQ accumulated Gram overflow')
            self._gram = updated
            self.tokens += values.shape[0]
            self.weighted_tokens += source_weight * values.shape[0]

    def take(self):
        with self._lock:
            if self._closed:
                raise RuntimeError('GSQ Gram capture is closed')
            if self._gram is None:
                raise ValueError('GSQ calibration has no positive-weight valid tokens')
            result = self._gram
            self._gram = None
            self._closed = True
            return result, {'tokens': self.tokens, 'weighted_tokens': self.weighted_tokens}

    def clear(self):
        """Release captured statistics and permanently close this collector."""
        with self._lock:
            self._gram = None
            self._closed = True
