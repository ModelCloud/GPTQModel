#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import site
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from llama_cpp import Llama
from llama_cpp import llama_cpp as llama_low
from transformers import AutoTokenizer

from gptqmodel import BACKEND, GGUFConfig, GPTQModel


os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:256")


DEFAULT_MODEL = "/monster/data/model/Llama-3.2-1B-Instruct"


@dataclass(frozen=True)
class TrialSummary:
    framework: str
    device: str
    phase: str
    token_count: int
    samples_ms: list[float]

    @property
    def mean_ms(self) -> float:
        return sum(self.samples_ms) / len(self.samples_ms)

    @property
    def min_ms(self) -> float:
        return min(self.samples_ms)

    @property
    def max_ms(self) -> float:
        return max(self.samples_ms)

    @property
    def toks_per_s(self) -> float:
        return self.token_count / (self.mean_ms / 1000.0)


def _ascii_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt(row: list[str]) -> str:
        return "| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) + " |"

    sep = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    out = [sep, fmt(headers), sep]
    for row in rows:
        out.append(fmt(row))
    out.append(sep)
    return "\n".join(out)


def _run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _sync_cuda(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _bench(fn: Callable[[], None], *, device: str, warmup: int, trials: int) -> list[float]:
    samples_ms: list[float] = []
    for _ in range(warmup):
        fn()
        _sync_cuda(device)

    for _ in range(trials):
        _sync_cuda(device)
        t0 = time.perf_counter()
        fn()
        _sync_cuda(device)
        samples_ms.append((time.perf_counter() - t0) * 1000.0)
    return samples_ms


def _find_convert_script() -> Path:
    for root in (Path(p) for p in site.getsitepackages()):
        candidate = root / "bin" / "convert_hf_to_gguf.py"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not locate convert_hf_to_gguf.py in site-packages.")


def _prepare_llama_cpp_monolithic(source_model: Path, f16_path: Path, q4_path: Path, threads: int) -> None:
    f16_path.parent.mkdir(parents=True, exist_ok=True)
    if not f16_path.exists():
        converter = _find_convert_script()
        _run(
            [
                "python",
                str(converter),
                str(source_model),
                "--outfile",
                str(f16_path),
                "--outtype",
                "f16",
            ]
        )

    if not q4_path.exists():
        params = llama_low.llama_model_quantize_default_params()
        params.ftype = llama_low.LLAMA_FTYPE_MOSTLY_Q4_K_M
        params.nthread = threads
        rc = llama_low.llama_model_quantize(
            str(f16_path).encode("utf-8"),
            str(q4_path).encode("utf-8"),
            params,
        )
        if rc != 0:
            raise RuntimeError(f"llama_model_quantize failed with status code {rc}.")


def _prepare_gptqmodel_quantized(source_model: Path, output_dir: Path, offload_dir: Path) -> None:
    if (output_dir / "quantize_config.json").exists():
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    offload_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(str(source_model), use_fast=True)
    qconfig = GGUFConfig(
        bits=4,
        format="q_k_m",
        smoother=None,
        offload_to_disk=True,
        offload_to_disk_path=str(offload_dir),
    )

    model = GPTQModel.from_pretrained(
        model_id_or_path=str(source_model),
        quantize_config=qconfig,
        trust_remote_code=False,
    )
    model.quantize(
        calibration=None,
        tokenizer=tokenizer,
        backend=BACKEND.GGUF_TORCH,
    )
    model.save(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _build_prompt(tokenizer: AutoTokenizer, target_tokens: int) -> tuple[str, int]:
    sentence = (
        "Summarize the scientific, historical, and economic significance of the Atlantic Ocean "
        "for intercontinental trade, climate, and biodiversity. "
    )
    prompt = sentence
    token_count = len(tokenizer(prompt, return_tensors="pt").input_ids[0])
    while token_count < target_tokens:
        prompt += sentence
        token_count = len(tokenizer(prompt, return_tensors="pt").input_ids[0])
    return prompt, token_count


def _load_gptqmodel(model_dir: Path, *, device: str):
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = GPTQModel.from_quantized(
        model_id_or_path=str(model_dir),
        backend=BACKEND.GGUF_TORCH,
        device="cuda:0" if device == "cuda" else "cpu",
        dtype=dtype,
        trust_remote_code=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    return model, tokenizer


def _load_llama_cpp(model_path: Path, *, device: str, n_ctx: int, n_batch: int, threads: int) -> Llama:
    kwargs = dict(
        model_path=str(model_path),
        n_ctx=n_ctx,
        n_batch=n_batch,
        n_ubatch=n_batch,
        n_threads=threads,
        n_threads_batch=threads,
        verbose=False,
        no_perf=True,
        use_mmap=True,
    )
    if device == "cuda":
        kwargs.update(
            {
                "n_gpu_layers": -1,
                "main_gpu": 0,
            }
        )
    else:
        kwargs.update({"n_gpu_layers": 0})
    return Llama(**kwargs)


def _gptq_prefill(model, tokenizer, prompt: str, device: str) -> tuple[list[float], int]:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    token_count = inputs["input_ids"].shape[1]

    def run_once() -> None:
        with torch.inference_mode():
            model.model(**inputs, use_cache=True)

    return run_once, token_count


def _gptq_decode(model, tokenizer, prompt: str, decode_tokens: int, device: str) -> tuple[Callable[[], None], int]:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    def run_once() -> None:
        with torch.inference_mode():
            out = model.model(**inputs, use_cache=True)
            past_key_values = out.past_key_values
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            for _ in range(decode_tokens):
                out = model.model(
                    input_ids=next_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = out.past_key_values
                next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    return run_once, decode_tokens


def _llama_prefill(llm: Llama, prompt: str) -> tuple[Callable[[], None], int]:
    tokens = llm.tokenize(prompt.encode("utf-8"), add_bos=True, special=False)

    def run_once() -> None:
        llm.reset()
        llm.eval(tokens)

    return run_once, len(tokens)


def _llama_decode(llm: Llama, prompt: str, decode_tokens: int) -> tuple[Callable[[], None], int]:
    tokens = llm.tokenize(prompt.encode("utf-8"), add_bos=True, special=False)

    def run_once() -> None:
        llm.reset()
        llm.eval(tokens)
        for _ in range(decode_tokens):
            token = llm.sample(temp=0.0, top_k=1, top_p=1.0, min_p=0.0)
            llm.eval([token])

    return run_once, decode_tokens


def _summarize_trials(framework: str, device: str, phase: str, token_count: int, samples_ms: list[float]) -> TrialSummary:
    return TrialSummary(
        framework=framework,
        device=device,
        phase=phase,
        token_count=token_count,
        samples_ms=samples_ms,
    )


def _print_trial_table(results: list[TrialSummary]) -> None:
    trial_count = max(len(r.samples_ms) for r in results)
    headers = ["framework", "phase", "tokens"] + [f"trial_{i}_ms" for i in range(1, trial_count + 1)]
    rows: list[list[str]] = []
    for result in results:
        row = [result.framework, result.phase, str(result.token_count)]
        row.extend(f"{sample:.2f}" for sample in result.samples_ms)
        if len(result.samples_ms) < trial_count:
            row.extend("-" for _ in range(trial_count - len(result.samples_ms)))
        rows.append(row)
    print(_ascii_table(headers, rows))


def _print_summary_table(results: list[TrialSummary]) -> None:
    headers = ["framework", "phase", "mean_ms", "min_ms", "max_ms", "tok_per_s"]
    rows = [
        [
            result.framework,
            result.phase,
            f"{result.mean_ms:.2f}",
            f"{result.min_ms:.2f}",
            f"{result.max_ms:.2f}",
            f"{result.toks_per_s:.2f}",
        ]
        for result in results
    ]
    print(_ascii_table(headers, rows))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark llama-cpp-python monolithic GGUF vs gptqmodel GGUF on the same model."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model directory to convert/quantize.")
    parser.add_argument("--work-dir", default="/tmp/llama_cpp_vs_gptqmodel_gguf", help="Artifact cache directory.")
    parser.add_argument("--prompt-tokens", type=int, default=512, help="Approximate prompt token length.")
    parser.add_argument("--decode-tokens", type=int, default=64, help="Number of autoregressive decode steps.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations per benchmark.")
    parser.add_argument("--trials", type=int, default=3, help="Measured trials per benchmark.")
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "both"),
        default="both",
        help="Run benchmarks on CPU, CUDA, or both.",
    )
    parser.add_argument("--threads", type=int, default=min(os.cpu_count() or 1, 16), help="CPU threads for llama.cpp.")
    parser.add_argument("--skip-prepare", action="store_true", help="Assume benchmark artifacts already exist.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_model = Path(args.model)
    work_dir = Path(args.work_dir)
    gptq_dir = work_dir / "gptqmodel_q4_k_m"
    offload_dir = work_dir / "gptqmodel_offload"
    llama_f16_path = work_dir / "llama3_2_1b_f16.gguf"
    llama_q4_path = work_dir / "llama3_2_1b_q4_k_m.gguf"

    if not args.skip_prepare:
        _prepare_llama_cpp_monolithic(source_model, llama_f16_path, llama_q4_path, args.threads)
        _prepare_gptqmodel_quantized(source_model, gptq_dir, offload_dir)

    tokenizer = AutoTokenizer.from_pretrained(str(source_model), use_fast=True)
    prompt, hf_token_count = _build_prompt(tokenizer, args.prompt_tokens)
    n_ctx = hf_token_count + args.decode_tokens + 64
    n_batch = max(hf_token_count + 8, 512)

    devices = ["cpu", "cuda"] if args.device == "both" else [args.device]
    if "cuda" in devices and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmarking requested but no CUDA device is available.")

    print(f"source_model={source_model}")
    print(f"gptqmodel_dir={gptq_dir}")
    print(f"llama_cpp_gguf={llama_q4_path}")
    print(f"prompt_tokens_hf={hf_token_count} decode_tokens={args.decode_tokens} warmup={args.warmup} trials={args.trials}")

    for device in devices:
        print()
        print(f"DEVICE {device}")

        gptq_model, gptq_tokenizer = _load_gptqmodel(gptq_dir, device=device)
        llama_model = _load_llama_cpp(
            llama_q4_path,
            device=device,
            n_ctx=n_ctx,
            n_batch=n_batch,
            threads=args.threads,
        )

        device_results: list[TrialSummary] = []

        gptq_prefill_fn, gptq_prefill_tokens = _gptq_prefill(gptq_model, gptq_tokenizer, prompt, device)
        gptq_decode_fn, gptq_decode_tokens = _gptq_decode(gptq_model, gptq_tokenizer, prompt, args.decode_tokens, device)
        llama_prefill_fn, llama_prefill_tokens = _llama_prefill(llama_model, prompt)
        llama_decode_fn, llama_decode_tokens = _llama_decode(llama_model, prompt, args.decode_tokens)

        device_results.append(
            _summarize_trials(
                "gptqmodel",
                device,
                "prefill",
                gptq_prefill_tokens,
                _bench(gptq_prefill_fn, device=device, warmup=args.warmup, trials=args.trials),
            )
        )
        device_results.append(
            _summarize_trials(
                "gptqmodel",
                device,
                "decode",
                gptq_decode_tokens,
                _bench(gptq_decode_fn, device=device, warmup=args.warmup, trials=args.trials),
            )
        )
        device_results.append(
            _summarize_trials(
                "llama-cpp-python",
                device,
                "prefill",
                llama_prefill_tokens,
                _bench(llama_prefill_fn, device="cpu", warmup=args.warmup, trials=args.trials),
            )
        )
        device_results.append(
            _summarize_trials(
                "llama-cpp-python",
                device,
                "decode",
                llama_decode_tokens,
                _bench(llama_decode_fn, device="cpu", warmup=args.warmup, trials=args.trials),
            )
        )

        _print_trial_table(device_results)
        _print_summary_table(device_results)

        del llama_model
        del gptq_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
