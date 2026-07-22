from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

import torch
from transformers import CompileConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from gptqmodel import BACKEND, GPTQModel  # noqa: E402
from gptqmodel.nn_modules.qlinear.gguf_triton import GGUFTritonKernel  # noqa: E402
from gptqmodel.utils.cuda_graph import StaticCUDAGraphGreedyRunner  # noqa: E402


DEFAULT_MODEL = Path("/monster/data/model/Ternary-Bonsai-1.7B-gguf/Ternary-Bonsai-1.7B-Q2_0.gguf")


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _release_q2_cache(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, GGUFTritonKernel):
            module.release_q2_prefill_cache()


def _measure_generate(
    generate: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iterations: int,
) -> tuple[list[float], list[float], torch.Tensor]:
    output = None
    for _ in range(warmup):
        output = generate()
    torch.cuda.synchronize()

    cuda_ms = []
    host_ms = []
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for index in range(iterations):
        host_start = time.perf_counter()
        starts[index].record()
        output = generate()
        ends[index].record()
        torch.cuda.synchronize()
        host_ms.append((time.perf_counter() - host_start) * 1000.0)
    cuda_ms.extend(start.elapsed_time(end) for start, end in zip(starts, ends))
    assert output is not None
    return cuda_ms, host_ms, output


def _print_result(
    label: str,
    cuda_ms: list[float],
    host_ms: list[float],
    *,
    new_tokens: int,
) -> None:
    median = statistics.median(cuda_ms)
    print(
        f"RESULT label={label} cuda_mean_ms={statistics.mean(cuda_ms):.4f} "
        f"cuda_p50_ms={median:.4f} cuda_p95_ms={_percentile(cuda_ms, 0.95):.4f} "
        f"host_p50_ms={statistics.median(host_ms):.4f} tok_s={new_tokens * 1000.0 / median:.2f}"
    )


def _print_memory(label: str) -> None:
    print(
        f"MEMORY label={label} allocated_mib={torch.cuda.memory_allocated() / (1024**2):.2f} "
        f"reserved_mib={torch.cuda.memory_reserved() / (1024**2):.2f} "
        f"peak_mib={torch.cuda.max_memory_allocated() / (1024**2):.2f}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Hugging Face Prism Q2_0 generation paths.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--prompt-tokens", type=int, default=64)
    parser.add_argument("--new-tokens", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument(
        "--paths",
        nargs="+",
        choices=("dynamic", "static", "compile", "graph"),
        default=("dynamic", "static", "graph"),
    )
    parser.add_argument("--graph-warmup", type=int, default=2)
    parser.add_argument("--max-cache-len", type=int)
    parser.add_argument("--retain-prefill-cache", action="store_true")
    parser.add_argument("--capture-prefill", action="store_true")
    parser.add_argument("--capture", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    torch.manual_seed(0)
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    properties = torch.cuda.get_device_properties(device)
    print(
        f"ENV torch={torch.__version__} cuda_runtime={torch.version.cuda} device={properties.name!r} "
        f"capability={properties.major}.{properties.minor} sms={properties.multi_processor_count} "
        f"memory_bytes={properties.total_memory}"
    )

    wrapper = GPTQModel.load(
        str(args.model),
        backend=BACKEND.GGUF_TRITON,
        profile="low_memory",
        device=device,
        dtype=torch.float16,
    )
    model = wrapper.model.eval()
    prompt = (torch.arange(args.prompt_tokens, device=device) % model.config.vocab_size).unsqueeze(0)
    common = {
        "do_sample": False,
        "eos_token_id": None,
        "max_new_tokens": args.new_tokens,
        "pad_token_id": model.config.eos_token_id,
        "use_cache": True,
    }
    outputs: dict[str, torch.Tensor] = {}
    generators: dict[str, Callable[[], torch.Tensor]] = {}

    with torch.inference_mode():
        for path in args.paths:
            torch.cuda.reset_peak_memory_stats()
            release_before_cold = True
            if path == "dynamic":
                generate = lambda: model.generate(  # noqa: E731
                    prompt,
                    cache_implementation="dynamic",
                    disable_compile=True,
                    **common,
                )
            elif path == "static":
                generate = lambda: model.generate(  # noqa: E731
                    prompt,
                    cache_implementation="static",
                    disable_compile=True,
                    **common,
                )
            elif path == "compile":
                compile_config = CompileConfig(fullgraph=False, dynamic=False, mode="reduce-overhead")
                generate = lambda: model.generate(  # noqa: E731
                    prompt,
                    cache_implementation="static",
                    compile_config=compile_config,
                    **common,
                )
            else:
                graph_capture_start = time.perf_counter()
                runner = StaticCUDAGraphGreedyRunner(
                    model,
                    prompt,
                    max_new_tokens=args.new_tokens,
                    max_cache_len=args.max_cache_len,
                    graph_warmup=args.graph_warmup,
                    release_prefill_cache=not args.retain_prefill_cache,
                    capture_prefill=args.capture_prefill,
                )
                release_before_cold = runner.release_prefill_cache
                torch.cuda.synchronize()
                print(
                    f"GRAPH_CAPTURE host_ms={(time.perf_counter() - graph_capture_start) * 1000.0:.4f} "
                    f"max_cache_len={runner.max_cache_len} release_prefill_cache={runner.release_prefill_cache} "
                    f"capture_prefill={runner.capture_prefill}"
                )
                generate = lambda runner=runner: runner.generate(  # noqa: E731
                    prompt,
                    max_new_tokens=args.new_tokens,
                )

            if release_before_cold:
                _release_q2_cache(model)
            torch.cuda.empty_cache()
            cold_start = time.perf_counter()
            cold_output = generate()
            torch.cuda.synchronize()
            print(f"COLD label={path} host_ms={(time.perf_counter() - cold_start) * 1000.0:.4f}")
            cuda_ms, host_ms, output = _measure_generate(
                generate,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            _print_result(path, cuda_ms, host_ms, new_tokens=args.new_tokens)
            _print_memory(path)
            print(
                f"CORRECTNESS label={path} cold_equal={bool(torch.equal(cold_output, output))} "
                f"shape={tuple(output.shape)} final_token={int(output[0, -1].item())}"
            )
            outputs[path] = output.clone()
            generators[path] = generate

    reference_label = next(iter(outputs))
    for label, output in outputs.items():
        print(
            f"CROSS_CORRECTNESS reference={reference_label} candidate={label} "
            f"equal={bool(torch.equal(outputs[reference_label], output))}"
        )

    if args.capture:
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        for label, generate in generators.items():
            torch.cuda.nvtx.range_push(f"prism_q2_generate_{label}")
            generate()
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
        torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
