"""Bounded natural-route activation capture for V4.1 main and draft blocks.

This captures raw FP32 X.T @ X sums. It does not normalize, regularize, augment
cold routes, quantize, or declare a projection durable. Those are explicit later
steps. Gate and up share one Hessian because their input tensors are identical.
"""

import torch


class V41Capture:
    def __init__(self, block, expert_ids, *, device, chunk_rows=1024, phase="both"):
        self.block = block
        self.expert_ids = tuple(expert_ids)
        experts = block.mlp.experts
        if (not self.expert_ids or len(set(self.expert_ids)) != len(self.expert_ids)
                or any(type(index) is not int or not 0 <= index < len(experts) for index in self.expert_ids)):
            raise ValueError("capture requires distinct valid expert IDs")
        if type(chunk_rows) is not int or chunk_rows < 1:
            raise ValueError("capture chunk size must be positive")
        self.device = torch.device(device)
        if phase not in ("both", "gate_up", "down"):
            raise ValueError("invalid projection capture phase")
        self.phase = phase
        self.chunk_rows = chunk_rows
        self.handles = []
        self.failed = False
        self.hessians = {}
        self.counts = {}
        self.route_counts = torch.zeros(len(experts), dtype=torch.int64)
        self.gate_squared_mass = torch.zeros(len(experts), dtype=torch.float64)
        for index in self.expert_ids:
            for family, width in (("gate_up", experts[index].gate_proj.in_features),
                                  ("down", experts[index].down_proj.in_features)):
                if phase != "both" and family != phase:
                    continue
                key = index, family
                self.hessians[key] = torch.zeros(width, width, device=device, dtype=torch.float32)
                self.counts[key] = 0

    @torch.no_grad()
    def _routes(self, module, args):
        _, indices, weights = args
        # Small route vectors on CPU give deterministic reduction order without
        # affecting dispatch, gate values, or the model's natural forward.
        indices = indices.detach().reshape(-1).cpu()
        weights = weights.detach().reshape(-1).cpu().double()
        if not torch.isfinite(weights).all():
            raise ValueError("nonfinite natural routing weights")
        self.route_counts += torch.bincount(indices, minlength=len(self.route_counts))
        self.gate_squared_mass.index_add_(0, indices, weights.square())

    @torch.no_grad()
    def _capture(self, key, args):
        inputs = args[0].detach().reshape(-1, args[0].shape[-1])
        hessian = self.hessians[key]
        if inputs.shape[-1] != hessian.shape[0]:
            raise ValueError("capture input width differs from projection")
        for start in range(0, inputs.shape[0], self.chunk_rows):
            rows = inputs[start:start + self.chunk_rows].to(device=self.device, dtype=torch.float32)
            if not torch.isfinite(rows).all():
                raise ValueError("nonfinite expert calibration input")
            hessian.addmm_(rows.T, rows)
        self.counts[key] += inputs.shape[0]

    def __enter__(self):
        if self.failed:
            raise RuntimeError("failed capture cannot be reused")
        if self.handles:
            raise RuntimeError("capture is already attached")
        if self.device.type == "cuda" and torch.backends.cuda.matmul.allow_tf32:
            raise ValueError("raw FP32 Hessian capture requires TF32 disabled before workers start")
        try:
            self.handles.append(self.block.mlp.experts.register_forward_pre_hook(self._routes))
            for index in self.expert_ids:
                expert = self.block.mlp.experts[index]
                for family, module in (("gate_up", expert.gate_proj), ("down", expert.down_proj)):
                    if (index, family) not in self.hessians:
                        continue
                    def hook(module, args, key=(index, family)):
                        self._capture(key, args)
                    self.handles.append(module.register_forward_pre_hook(hook))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    @torch.no_grad()
    def capture_routed(self, batch):
        """Capture a precomputed routed batch without replaying other experts.

        Gate/up needs no expert GEMM at all. Down runs only selected experts'
        gate/up and forms their route-weighted activation; no down GEMM is needed.
        The router batch must come from this block's immutable input frontier.
        """
        if self.handles or self.failed:
            raise RuntimeError("direct capture must be detached and healthy")
        if self.device.type == "cuda" and torch.backends.cuda.matmul.allow_tf32:
            raise ValueError("direct capture requires TF32 disabled")
        try:
            batch.validate()
            self._routes(None, (batch.hidden, batch.indices, batch.weights))
            for index in self.expert_ids:
                tokens, slots = torch.where(batch.indices == index)
                if not tokens.numel():
                    continue
                inputs = batch.hidden[tokens].to(self.device)
                if (index, "gate_up") in self.hessians:
                    self._capture((index, "gate_up"), (inputs,))
                if (index, "down") in self.hessians:
                    expert = self.block.mlp.experts[index]
                    gate, up = expert.gate_proj(inputs).float(), expert.up_proj(inputs).float()
                    if expert.limit > 0:
                        gate = gate.clamp(max=expert.limit)
                        up = up.clamp(min=-expert.limit, max=expert.limit)
                    weights = batch.weights[tokens, slots].to(self.device)
                    intermediate = (torch.nn.functional.silu(gate) * up * weights[:, None]).to(inputs.dtype)
                    self._capture((index, "down"), (intermediate,))
        except BaseException:
            self.failed = True
            raise

    def __exit__(self, *exc):
        if exc and exc[0] is not None:
            self.failed = True
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def projection(self, expert, projection):
        """Return owned, unfinalized CPU Hessian and natural route evidence."""
        if self.failed:
            raise RuntimeError("failed capture cannot export partial evidence")
        if self.handles:
            raise RuntimeError("detach capture before exporting evidence")
        family = {"w1": "gate_up", "w3": "gate_up", "w2": "down"}[projection]
        key = expert, family
        count = self.counts[key]
        if count != int(self.route_counts[expert]):
            raise ValueError("projection rows differ from natural dispatch count")
        mass = self.gate_squared_mass.sum().item()
        return {"H": self.hessians[key].detach().cpu().clone(), "count": count, "finalized": False}, {
            "natural_rows": count, "expert_gate_squared_mass": self.gate_squared_mass[expert].item(),
            "total_gate_squared_mass": mass,
            "expert_gate_squared_mass_fraction": self.gate_squared_mass[expert].item() / mass if mass else 0.0}
