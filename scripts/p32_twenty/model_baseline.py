"""Bounded real-model baselines; all results are written outside the fixed snapshot."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SNAPSHOT = Path(
    "/root/qvq-results/calibration-fisher-frontier-wave14-v1/llama32-1b-f6_yaqa125x_seed7"
)
DENSE = "/monster/data/model/Llama-3.2-1B-Instruct"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("bf16", "canonical", "window", "production"), required=True
    )
    parser.add_argument("--capture-only", action="store_true")
    parser.add_argument("--uuid", required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(SNAPSHOT.resolve()):
        parser.error("Output must be outside the snapshot")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.uuid
    for sample in range(3):
        text = subprocess.check_output(
            [
                "nvidia-smi",
                "--id=" + args.uuid,
                "--query-gpu=uuid,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        fields = [s.strip() for s in text.split(",")]
        if fields[0] != args.uuid or int(fields[1]) > 8 or int(fields[2]) != 0:
            raise RuntimeError("Idle gate failed: " + text)
        foreign = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        if args.uuid in foreign:
            raise RuntimeError("Foreign process on target GPU")
        print("IDLE", sample + 1, text, flush=True)
        time.sleep(1)
    import torch
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.set_float32_matmul_precision("highest")
    args.output.mkdir(parents=True, exist_ok=True)
    inputs = json.loads(args.inputs.read_text())
    report = {
        "mode": args.mode,
        "uuid": args.uuid,
        "torch": torch.__version__,
        "device": str(torch.cuda.get_device_properties(0)),
        "snapshot": str(SNAPSHOT),
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "scope": "bounded model baseline, not complete experiment scorecard",
        "quality": [],
        "performance": [],
    }
    if args.mode == "production":
        from gptqmodel import BACKEND, GPTQModel

        wrapper = GPTQModel.load(
            str(SNAPSHOT),
            backend=BACKEND.QVQ,
            dtype=torch.float16,
            device_map={"": "cuda:0"},
            attn_implementation="eager",
        )
        model = wrapper.model.eval()
    else:
        dtype = torch.bfloat16 if args.mode == "bf16" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            DENSE,
            dtype=dtype,
            device_map={"": "cuda:0"},
            attn_implementation="eager",
            local_files_only=True,
        ).eval()
        if args.mode != "bf16":
            from gptqmodel.quantization.qvq import (
                reconstruct_p32_window_inner_weight,
                reconstruct_qvq_inner_weight,
                repack_p32_planar_to_window,
            )
            from gptqmodel.quantization.rotation.hadamard_utils import matmul_hadU

            index = json.loads((SNAPSHOT / "model.safetensors.index.json").read_text())[
                "weight_map"
            ]

            def read(name):
                with safe_open(
                    str(SNAPSHOT / index[name]), framework="pt", device="cpu"
                ) as f:
                    return f.get_tensor(name).to("cuda")

            # Copy every saved dense parameter so the oracle represents this snapshot, including norms/embedding.
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in index:
                        param.copy_(read(name))

            class CanonicalLinear(torch.nn.Module):
                def __init__(self, inner, su, sv):
                    super().__init__()
                    self.register_buffer("inner", inner.float())
                    self.register_buffer("SU", su.float())
                    self.register_buffer("SV", sv.float())

                def forward(self, x):
                    return (
                        matmul_hadU(matmul_hadU(x.float() * self.SU) @ self.inner)
                        * self.SV
                    )

            # This historical config uses RHT on both sides and no activation quantization or projection bias.
            cfg = json.loads((SNAPSHOT / "quantize_config.json").read_text())
            if cfg.get("incoherence") != "rht" or cfg.get("activation"):
                raise RuntimeError(
                    "Canonical implementation requires the pinned RHT/no-activation contract"
                )
            for name in sorted(index):
                if not name.endswith(".trellis"):
                    continue
                prefix = name[:-8]
                trellis, su, sv = (
                    read(prefix + "." + k) for k in ("trellis", "SU", "SV")
                )
                p32 = prefix + ".bank_alt_id" in index
                bits = trellis.shape[-1] / 8
                bank = (
                    read(prefix + ".bank_ids")
                    if prefix + ".bank_ids" in index
                    else None
                )
                alt = read(prefix + ".bank_alt_id") if p32 else None
                kw = {
                    "bits": bits,
                    "in_features": su.numel(),
                    "out_features": sv.numel(),
                    "bank_ids": bank,
                    "bank_alt_id": alt,
                    "codebook_version": cfg["codebook"],
                }
                if args.mode == "window" and p32:
                    inner = reconstruct_p32_window_inner_weight(
                        repack_p32_planar_to_window(trellis, bits=bits), **kw
                    )
                else:
                    inner = reconstruct_qvq_inner_weight(trellis, v2b2_p32=p32, **kw)
                parent, leaf = prefix.rsplit(".", 1)
                setattr(
                    model.get_submodule(parent), leaf, CanonicalLinear(inner, su, sv)
                )
                print("RECONSTRUCTED", prefix, flush=True)
            del trellis, su, sv, bank, alt, inner
    if args.capture_only:
        if args.mode != "canonical":
            raise ValueError("Capture requires canonical teacher")
        captured = []
        hooks = []

        def hook(name):
            def save(module, values, output):
                path = args.output / (name + ".pt")
                torch.save(
                    {
                        "input": values[0].detach().cpu(),
                        "teacher_output": output.detach().cpu(),
                    },
                    path,
                )
                captured.append(
                    {"module": name, "path": str(path), "shape": list(values[0].shape)}
                )

            return save

        for name, module in model.named_modules():
            if isinstance(module, CanonicalLinear) and name.split(".")[2] in ("0", "1"):
                hooks.append(module.register_forward_hook(hook(name)))
        ids = [token for row in inputs["rows"] for token in row["input_ids"]][:2048]
        with torch.inference_mode():
            model(torch.tensor([ids], device="cuda"), use_cache=False)
        for handle in hooks:
            handle.remove()
        (args.output / "capture.json").write_text(
            json.dumps(
                {
                    "scope": "timing/kernel correctness only; not calibration for fitting",
                    "modules": captured,
                },
                indent=2,
            )
            + "\n"
        )
        print("CAPTURED", len(captured), flush=True)
        return
    with torch.inference_mode():
        total_nll = 0.0
        total_tokens = 0
        for j, row in enumerate(inputs["rows"]):
            ids = torch.tensor([row["input_ids"]], device="cuda")
            logits = model(ids, use_cache=False).logits.float()
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError("Non-finite logits")
            nll = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                ids[:, 1:].reshape(-1),
                reduction="sum",
            ).item()
            count = ids.numel() - 1
            total_nll += nll
            total_tokens += count
            torch.save(logits.cpu(), args.output / f"logits-{j:03d}.pt")
            report["quality"].append({"row": j, "nll_sum": nll, "tokens": count})
            report["perplexity"] = float(
                torch.tensor(total_nll / total_tokens, dtype=torch.float64).exp()
            )
            (args.output / "report.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            print(
                "QUALITY",
                j + 1,
                len(inputs["rows"]),
                "PPL",
                report["perplexity"],
                flush=True,
            )
        flat = [token for row in inputs["rows"] for token in row["input_ids"]]
        for m in (1, 2, 4, 8, 16, 32, 128, 512, 2048):
            ids = torch.tensor([flat[:m]], device="cuda")
            fn = lambda ids=ids: model(ids, use_cache=False)
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            processes = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-compute-apps=gpu_uuid,pid",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            )
            for line in processes.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if parts[0] == args.uuid and int(parts[1]) != os.getpid():
                    raise RuntimeError("Foreign process appeared before timing")
            samples = []
            for _ in range(10):
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                out = fn()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end))
                del out
            report["performance"].append(
                {"regime": "prefill", "tokens": m, "samples_ms": samples}
            )
            print("PREFILL", m, "ms", sorted(samples)[5], flush=True)
        ids = torch.tensor([flat[:128]], device="cuda")
        out = model(ids, use_cache=True)
        cache = out.past_key_values
        token = out.logits[:, -1:].argmax(-1)
        decode = []
        for _ in range(32):
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            out = model(token, past_key_values=cache, use_cache=True)
            end.record()
            end.synchronize()
            decode.append(start.elapsed_time(end))
            cache = out.past_key_values
            token = out.logits[:, -1:].argmax(-1)
        report["performance"].append(
            {
                "regime": "decode",
                "prompt_tokens": 128,
                "new_tokens": 32,
                "samples_ms": decode,
                "note": "growing KV cache, cold first decode included; no CUDA graphs",
            }
        )
    report["complete"] = True
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
