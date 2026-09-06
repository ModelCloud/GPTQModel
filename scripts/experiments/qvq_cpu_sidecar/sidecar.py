import os
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

from .inventory import cpu_runtime_inventory

capture_lock = threading.Lock()
_worker_cache = None


class FactorSidecar:
    def __init__(self, modules, sequences, rank, seed, cls):
        global _worker_cache
        self.modules = modules
        self.sequences = float(sequences)
        self.rank = rank
        self.seed = seed
        self.cls = cls
        self.jobs = deque()
        self.factors = {}
        self.counter = 0
        self.module_jobs = 0
        self.records = []
        key = tuple(sorted(os.sched_getaffinity(0)))
        if _worker_cache is None or _worker_cache[0] != key:
            if _worker_cache is not None:
                for pool in _worker_cache[1]:
                    pool.shutdown(wait=True)
            inv = cpu_runtime_inventory()
            cores = []
            for node in range(2):
                allowed = set(inv["numa_allowed_cpus"][f"node{node}"])
                cores.append(
                    next(x["allowed_cpus"][0] for x in inv["physical_core_groups"] if set(x["allowed_cpus"]) & allowed)
                )
            _worker_cache = (key, [ThreadPoolExecutor(max_workers=1) for _ in range(2)], threading.local(), cores)
        self.pools, self.local, self.cores = _worker_cache[1:]

    def run(self, slot, names, tensors, event):
        if not hasattr(self.local, "ready"):
            os.sched_setaffinity(0, {self.cores[slot]})
            self.local.ready = True
        start = time.perf_counter()
        event.synchronize()
        ready = time.perf_counter()
        sa, sd, na, nd, da, dd = tensors
        result = []
        for j, name in enumerate(names):
            module = self.modules[name]
            pair = []
            for source, diag, norm, width in [
                (sa[j], da[j], na[j], module.out_features),
                (sd[j], dd[j], nd[j], module.in_features),
            ]:
                denominator = self.sequences * width
                pair.append(
                    self.cls(
                        source=source.contiguous(),
                        diagonal=diag.div(denominator).clamp_min_(0).contiguous(),
                        normalizer=denominator * self.rank,
                        seed=self.seed,
                        source_diagonal=norm.contiguous(),
                        _source_diagonal_validated=True,
                        _finite_nonnegative_validated=denominator >= 1,
                    )
                )
            result.append((name, tuple(pair)))
        self.records.append((ready - start, time.perf_counter() - ready))
        return result

    def submit(self, names, tensors, event):
        self.drain(False)
        while len(self.jobs) >= 16:
            self.drain_one()
        slot = self.counter % 2
        self.counter += 1
        self.module_jobs += len(names)
        self.jobs.append(self.pools[slot].submit(self.run, slot, names, tensors, event))

    def drain_one(self):
        self.factors.update(self.jobs.popleft().result())

    def drain(self, all):
        while self.jobs and (all or self.jobs[0].done()):
            self.drain_one()

    def finish(self):
        errors = []
        while self.jobs:
            try:
                self.drain_one()
            except BaseException as error:  # noqa: BLE001 - drain queued CUDA-backed jobs before rethrowing
                errors.append(error)
        if errors:
            raise errors[0]
        print(
            {
                "cpu_factor_modules": self.module_jobs,
                "event_wait_sum": sum(x[0] for x in self.records),
                "compute_sum": sum(x[1] for x in self.records),
            },
            flush=True,
        )


def close_workers():
    global _worker_cache
    if _worker_cache is not None:
        for pool in _worker_cache[1]:
            pool.shutdown(wait=True)
        _worker_cache = None
