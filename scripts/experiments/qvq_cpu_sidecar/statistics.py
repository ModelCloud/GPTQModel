import torch


class GroupStatistics:
    def __init__(self, raw_a, raw_d, mask, rank, project_module, pool=None, stream=None):
        self.count, self.batch, self.tokens, self.inputs = raw_a.shape
        self.outputs = raw_d.shape[-1]
        self.rank = rank
        self.project_module = project_module
        self.raw_a = torch.empty_like(raw_a)
        self.raw_d = torch.empty_like(raw_d)
        self.mask = torch.empty_like(mask)
        self.source_a = torch.zeros(self.count, self.inputs, rank, dtype=torch.float32, device=raw_a.device)
        self.source_d = torch.zeros(self.count, self.outputs, rank, dtype=torch.float32, device=raw_a.device)
        self.diag_a = torch.zeros(self.count, self.inputs, dtype=torch.float32, device=raw_a.device)
        self.diag_d = torch.zeros(self.count, self.outputs, dtype=torch.float32, device=raw_a.device)
        self.generators = [torch.Generator(device=raw_a.device).manual_seed(i + 1) for i in range(self.count)]
        self.graph = torch.cuda.CUDAGraph()
        stream = stream or torch.cuda.Stream(device=raw_a.device)
        stream.wait_stream(torch.cuda.current_stream(raw_a.device))
        with torch.cuda.stream(stream):
            self.raw_a.copy_(raw_a)
            self.raw_d.copy_(raw_d)
            self.mask.copy_(mask)
            # Warm library dispatch before capture, then discard that accumulation.
            self._core()
            self.reset()
            self.graph.capture_begin(pool=pool, capture_error_mode="thread_local")
            try:
                self.result = self._core()
            finally:
                self.graph.capture_end()
            self.norm_graph = torch.cuda.CUDAGraph()
            self._norm_core()
            self.norm_graph.capture_begin(pool=pool, capture_error_mode="thread_local")
            try:
                self.norm_result = self._norm_core()
            finally:
                self.norm_graph.capture_end()
        torch.cuda.current_stream(raw_a.device).wait_stream(stream)

    def reset(self):
        self.source_a.zero_()
        self.source_d.zero_()
        self.diag_a.zero_()
        self.diag_d.zero_()

    def _project(self, x, p):
        k = x.shape[-1]
        if k in (5120, 17408):
            bm, bn = (32, 64)
            out = torch.empty(self.count * self.batch, self.tokens, self.rank, dtype=torch.float32, device=x.device)
            self.project_module._project_kernel[(self.tokens // bm, self.rank // bn, self.count * self.batch)](
                x, p, out, k, *x.stride()[:2], *p.stride()[:2], bm, bn, num_warps=4
            )
            return out
        return torch.bmm(x, p)

    def _core(self):
        m, b, t, i = self.count, self.batch, self.tokens, self.inputs
        o = self.outputs
        r = self.rank
        keep = self.mask[None, :, :, None]
        narrow_a = torch.where(keep, self.raw_a, 0)
        narrow_d = torch.where(keep, self.raw_d, 0)
        a = narrow_a.float().reshape(m * b, t, i)
        d = narrow_d.float().reshape(m * b, t, o)
        projection = torch.empty(m, b, o + i, r, dtype=torch.float32, device=a.device)
        for index, generator in enumerate(self.generators):
            projection[index].normal_(generator=generator)
        projection = projection.reshape(m * b, o + i, r)
        si = torch.bmm(a.transpose(1, 2), self._project(d, projection[:, :o])).reshape(m, b, i, r).sum(1)
        so = torch.bmm(d.transpose(1, 2), self._project(a, projection[:, o:])).reshape(m, b, o, r).sum(1)
        self.source_a.add_(si)
        self.source_d.add_(so)
        ga = torch.bmm(a, a.transpose(1, 2))
        gd = torch.bmm(d, d.transpose(1, 2))
        di = (a * torch.bmm(gd, a)).reshape(m, b, t, i)
        do = (d * torch.bmm(ga, d)).reshape(m, b, t, o)
        self.diag_a.add_(torch.stack([x.sum((0, 1)) for x in di.unbind(0)]))
        self.diag_d.add_(torch.stack([x.sum((0, 1)) for x in do.unbind(0)]))
        return self.diag_a, self.diag_d, self.source_a, self.source_d

    def _norm_core(self):
        source_diag_a = torch.stack([x.square().sum(1) for x in self.source_a.unbind(0)])
        source_diag_d = torch.stack([x.square().sum(1) for x in self.source_d.unbind(0)])
        bad = (~torch.isfinite(source_diag_a).all()) | (~torch.isfinite(source_diag_d).all())
        return source_diag_a, source_diag_d, bad

    def __call__(self, raw_a, raw_d, mask, seeds, last=True):
        self.raw_a.copy_(raw_a)
        self.raw_d.copy_(raw_d)
        self.mask.copy_(mask)
        for generator, seed in zip(self.generators, seeds):
            generator.manual_seed(seed)
        self.graph.replay()
        if last:
            self.norm_graph.replay()
        return (*self.result, *self.norm_result)
