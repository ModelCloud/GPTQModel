"""Experimental standalone W4A16 + FP32 low-rank operator loader.

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
        self.eval()

    def forward(self, x):
        shape = x.shape[:-1]
        xf = x.reshape(-1, self.in_features).float()
        y = self.base(xf.bfloat16()).float() + (xf @ self.a) @ self.b
        return y.to(x.dtype).reshape(*shape, self.out_features)
