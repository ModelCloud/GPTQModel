"""Experimental standalone W4A16 + typed low-rank operator loader.

Only load exports produced by this campaign: torch.load uses trusted pickle data.
Output preserves the caller's dtype; full-model FP16 boundaries need own validation.
"""

import torch
import torchao  # noqa: F401 -- register exported tensor classes


class RecoveredLinear(torch.nn.Module):
    def __init__(self, path, device="cuda"):
        super().__init__()
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        self.in_features = bundle["in_features"]
        self.out_features = bundle["out_features"]
        self.source_module = bundle["module"]
        self.base = torch.nn.Linear(
            self.in_features,
            self.out_features,
            bias=False,
            device=device,
            dtype=torch.bfloat16,
        )
        self.base.weight = torch.nn.Parameter(
            bundle["native_state_dict"]["weight"].to(device), requires_grad=False
        )
        self.register_buffer("a", bundle["a"].to(device))
        self.register_buffer("b", bundle["b"].to(device))
        self.register_buffer(
            "sparse_input_indices",
            bundle.get("sparse_input_indices", torch.empty(0, dtype=torch.int32)).to(
                device
            ),
        )
        self.register_buffer(
            "sparse_output_indices",
            bundle.get("sparse_output_indices", torch.empty(0, dtype=torch.int32)).to(
                device
            ),
        )
        self.register_buffer(
            "sparse_values",
            bundle.get("sparse_values", torch.empty(0, dtype=torch.float32)).to(device),
        )
        self.sparse_nnz = self.sparse_values.numel()
        self.eval()

    def forward(self, x):
        shape = x.shape[:-1]
        xf = x.reshape(-1, self.in_features).float()
        y = self.base(xf.bfloat16()).float()
        if self.a.shape[1]:
            hidden = xf.to(self.a.dtype) @ self.a
            correction = hidden.to(self.b.dtype) @ self.b
            y = y + correction.float()
        if self.sparse_nnz:
            updates = xf[:, self.sparse_input_indices.long()] * self.sparse_values
            y.index_add_(1, self.sparse_output_indices.long(), updates)
        return y.to(x.dtype).reshape(*shape, self.out_features)
