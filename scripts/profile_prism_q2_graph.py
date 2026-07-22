from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch
from transformers import StaticCache


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from gptqmodel import BACKEND, GPTQModel  # noqa: E402
from gptqmodel.nn_modules.qlinear.gguf_triton import GGUFTritonKernel  # noqa: E402


DEFAULT_MODEL = Path("/monster/data/model/Ternary-Bonsai-1.7B-gguf/Ternary-Bonsai-1.7B-Q2_0.gguf")


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _measure(fn, *, warmup: int, iterations: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for index in range(iterations):
        starts[index].record()
        fn()
        ends[index].record()
    torch.cuda.synchronize()
    return [start.elapsed_time(end) for start, end in zip(starts, ends)]


def _print_result(label: str, values: list[float], *, tokens: int = 1) -> None:
    print(
        f"RESULT label={label} mean_ms={statistics.mean(values):.4f} "
        f"p50_ms={statistics.median(values):.4f} p95_ms={_percentile(values, 0.95):.4f} "
        f"min_ms={min(values):.4f} max_ms={max(values):.4f} "
        f"tok_s={tokens * 1000.0 / statistics.median(values):.2f}"
    )


def _q2_modules(model: torch.nn.Module) -> list[GGUFTritonKernel]:
    return [module for module in model.modules() if isinstance(module, GGUFTritonKernel)]


def _release_q2_cache(model: torch.nn.Module) -> None:
    for module in _q2_modules(model):
        module.release_q2_prefill_cache()


def _dynamic_decode(model: torch.nn.Module, prompt: torch.Tensor):
    output = model(input_ids=prompt, use_cache=True, logits_to_keep=1)
    cache = output.past_key_values
    token = output.logits[:, -1:, :].argmax(dim=-1)

    def step():
        nonlocal cache, token
        output = model(input_ids=token, past_key_values=cache, use_cache=True, logits_to_keep=1)
        cache = output.past_key_values
        token = output.logits[:, -1:, :].argmax(dim=-1)
        return output

    return step


def _static_decode(model: torch.nn.Module, prompt: torch.Tensor, *, max_cache_len: int):
    cache = StaticCache(config=model.config, max_cache_len=max_cache_len)
    positions = torch.arange(prompt.shape[1], device=prompt.device).unsqueeze(0)
    output = model(
        input_ids=prompt,
        position_ids=positions,
        past_key_values=cache,
        use_cache=True,
        logits_to_keep=1,
    )
    token = output.logits[:, -1:, :].argmax(dim=-1)
    position = torch.full((1, 1), prompt.shape[1], device=prompt.device, dtype=torch.long)

    def step():
        nonlocal token
        output = model(
            input_ids=token,
            position_ids=position,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
        token = output.logits[:, -1:, :].argmax(dim=-1)
        position.add_(1)
        return output

    return step, lambda: token


def _graph_decode(model: torch.nn.Module, prompt: torch.Tensor, *, max_cache_len: int, graph_warmup: int):
    cache = StaticCache(config=model.config, max_cache_len=max_cache_len)
    positions = torch.arange(prompt.shape[1], device=prompt.device).unsqueeze(0)
    output = model(
        input_ids=prompt,
        position_ids=positions,
        past_key_values=cache,
        use_cache=True,
        logits_to_keep=1,
    )
    token = output.logits[:, -1:, :].argmax(dim=-1).contiguous()
    position = torch.full((1, 1), prompt.shape[1], device=prompt.device, dtype=torch.long)

    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        for _ in range(graph_warmup):
            output = model(
                input_ids=token,
                position_ids=position,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            token.copy_(output.logits[:, -1:, :].argmax(dim=-1))
            position.add_(1)
    torch.cuda.current_stream().wait_stream(capture_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        graph_output = model(
            input_ids=token,
            position_ids=position,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
        token.copy_(graph_output.logits[:, -1:, :].argmax(dim=-1))
        position.add_(1)
    torch.cuda.synchronize()
    graph_state = (graph, cache, token, position, graph_output)

    def replay():
        graph.replay()
        return graph_state[-1]

    return replay, lambda: token, lambda: int(cache.get_seq_length().item()), graph_warmup


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile eager and CUDA Graph Prism Q2_0 decode paths.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--prompt-tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--max-cache-len", type=int, default=256)
    parser.add_argument("--graph-warmup", type=int, default=2)
    parser.add_argument("--capture", action="store_true")
    parser.add_argument("--profile-dynamic", type=int, default=4)
    parser.add_argument("--profile-graph", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    required_cache_len = args.prompt_tokens + args.graph_warmup + args.iterations
    if args.capture:
        required_cache_len += args.profile_graph
    if args.max_cache_len < required_cache_len:
        raise ValueError(
            f"--max-cache-len must be at least {required_cache_len} for the requested graph replays; "
            f"actual value is {args.max_cache_len}."
        )

    torch.manual_seed(0)
    torch.cuda.set_device(args.device)
    wrapper = GPTQModel.load(
        str(args.model),
        backend=BACKEND.GGUF_TRITON,
        profile="low_memory",
        device=f"cuda:{args.device}",
        dtype=torch.float16,
    )
    model = wrapper.model.eval()
    prompt = (
        torch.arange(args.prompt_tokens, device=f"cuda:{args.device}", dtype=torch.long) % model.config.vocab_size
    ).unsqueeze(0)

    with torch.inference_mode():
        fused_rms_norms = [
            module for module in model.modules() if getattr(module, "_gptqmodel_prism_q2_rms_norm", False)
        ]
        fused_swiglu = [
            module for module in model.modules() if getattr(module, "_gptqmodel_prism_q2_swiglu", False)
        ]
        fused_qkv = [
            module for module in model.modules() if getattr(module, "_gptqmodel_prism_q2_qkv", False)
        ]
        print(
            f"FUSION installed_rms_norms={len(fused_rms_norms)} "
            f"installed_swiglu={len(fused_swiglu)} installed_qkv={len(fused_qkv)}"
        )

        def prefill():
            return model(input_ids=prompt, use_cache=True)

        prefill_values = _measure(prefill, warmup=args.warmup, iterations=args.iterations)
        _print_result("prefill_fused_rms", prefill_values, tokens=args.prompt_tokens)

        dynamic_step = _dynamic_decode(model, prompt)
        dynamic = _measure(dynamic_step, warmup=args.warmup, iterations=args.iterations)
        _print_result("dynamic_fused_decode", dynamic)

        static_step, _ = _static_decode(model, prompt, max_cache_len=args.max_cache_len)
        static = _measure(static_step, warmup=args.warmup, iterations=args.iterations)
        _print_result("static_fused_decode", static)

        _release_q2_cache(model)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        graph_step, graph_token, graph_cache_length, graph_prior_steps = _graph_decode(
            model,
            prompt,
            max_cache_len=args.max_cache_len,
            graph_warmup=args.graph_warmup,
        )
        graph = _measure(graph_step, warmup=0, iterations=args.iterations)
        _print_result("graph_static_fused_decode", graph)
        graph_final_token = graph_token().clone()
        graph_length_after_benchmark = graph_cache_length()
        allocated = torch.cuda.memory_allocated() / (1024**2)
        reserved = torch.cuda.memory_reserved() / (1024**2)
        peak = torch.cuda.max_memory_allocated() / (1024**2)

        eager_reference_step, eager_reference_token = _static_decode(
            model,
            prompt,
            max_cache_len=args.max_cache_len,
        )
        for _ in range(graph_prior_steps + args.iterations):
            eager_reference_step()
        torch.cuda.synchronize()
        print(
            f"GRAPH_CORRECTNESS final_token_equal="
            f"{bool(torch.equal(graph_final_token, eager_reference_token()))} "
            f"graph_token={graph_final_token.item()} eager_token={eager_reference_token().item()} "
            f"graph_cache_length={graph_length_after_benchmark}"
        )
        print(f"MEMORY graph_allocated_mib={allocated:.2f} graph_reserved_mib={reserved:.2f} graph_peak_mib={peak:.2f}")

        if args.capture:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            torch.cuda.nvtx.range_push("prism_q2_dynamic_fused_decode")
            for _ in range(args.profile_dynamic):
                dynamic_step()
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
            torch.cuda.nvtx.range_push("prism_q2_graph_fused_decode")
            for _ in range(args.profile_graph):
                graph_step()
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
            torch.cuda.cudart().cudaProfilerStop()
            print(f"CAPTURE graph_cache_length={graph_cache_length()}")


if __name__ == "__main__":
    main()
