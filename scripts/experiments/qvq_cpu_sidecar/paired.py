"""Benchmark-only alternation: CPU finalization, accepted GPU, grouped GPU."""

import os

import torch

from scripts.experiments.qvq_cpu_sidecar import collector

YaqaGramSketch = collector.YaqaGramSketch
_counter = 0


def capture_yaqa_sketch_b(*args, **kwargs):
    global _counter
    arm = ("cpu_finalize", "current_gpu", "grouped_gpu")[_counter % 3]
    _counter += 1
    if arm == "current_gpu":
        result = collector._baseline.capture_yaqa_sketch_b(*args, **kwargs)
    else:
        previous = os.environ.get("QVQ_CPU_FINALIZE")
        os.environ["QVQ_CPU_FINALIZE"] = "1" if arm == "cpu_finalize" else "0"
        try:
            result = collector.capture_yaqa_sketch_b(*args, **kwargs)
        finally:
            if previous is None:
                os.environ.pop("QVQ_CPU_FINALIZE", None)
            else:
                os.environ["QVQ_CPU_FINALIZE"] = previous
        expected = "factor_finalize" if arm == "cpu_finalize" else "disabled_grouped_gpu"
        if result[2].get("cpu_sidecar") != expected:
            raise RuntimeError("Experimental collector is unsupported in this configuration")
    result[2]["comparison_arm"] = arm
    result[2]["host_pinned_allocated_bytes"] = torch.cuda.memory.host_memory_stats()["allocated_bytes.current"]
    return result
