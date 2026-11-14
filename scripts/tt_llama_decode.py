# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Serve, benchmark or check an out-axis block-float Llama on Tenstorrent.

Usage::

    # chat generation on card 0
    python scripts/tt_llama_decode.py generate CKPT --prompt "What is 17 * 23?"
    # traced decode throughput at a context position, one process per card
    python scripts/tt_llama_decode.py bench CKPT --pos 512 --cards 4
    # teacher-forced top-1 agreement with the CPU fp32 model on the same weights
    python scripts/tt_llama_decode.py parity CKPT

``CKPT`` must be a tt-blockfloat checkpoint written with ``block_axis="out"``
(``scripts/tt_quantize_blockfloat.py --block-axis out``).

Multi-card throughput is data parallel with one process per card: each child
sees a single chip (``TT_VISIBLE_DEVICES``) and opens it with the single-chip
fabric descriptor, which measured linear scaling where a 1xN ``MeshDevice``
running the same trace was ~2x slower per step.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time


TRACE_REGION = 200_000_000


def _single_chip_descriptor() -> str | None:
    try:
        import pjrt_plugin_tt
    except Exception:
        return None
    root = os.path.join(os.path.dirname(pjrt_plugin_tt.__file__), "tt-metal")
    hits = glob.glob(os.path.join(root, "tt_metal", "fabric", "mesh_graph_descriptors", "p150_mesh_graph_descriptor.textproto"))
    return hits[0] if hits else None


def _open(args):
    from gptqmodel.utils.ttnn_runtime import open_mesh

    return open_mesh(device_count=1, trace_region_size=TRACE_REGION, owner="tt_llama_decode")


def _decoder(args, mesh):
    from gptqmodel.utils.tt_llama import TTLlamaDecoder, load_tt_llama_weights

    config, weights = load_tt_llama_weights(args.checkpoint)
    t0 = time.time()
    decoder = TTLlamaDecoder(mesh.device, config, weights, max_seq=args.max_seq, fidelity=args.fidelity,
                             kv_dtype=args.kv_dtype, check_drift=not args.skip_drift)
    drift = max(decoder.drift.values()) if decoder.drift else float("nan")
    print(f"[TT] uploaded {len(decoder.drift)} block-float tensors in {time.time() - t0:.1f}s, "
          f"max drift {drift:.1e}", flush=True)
    if decoder.drift and drift != 0.0:
        print("[TT] ERROR: block-float upload is not exact; is this an out-axis checkpoint?", flush=True)
        raise SystemExit(1)
    return config, decoder


def cmd_generate(args):
    from transformers import AutoTokenizer

    from gptqmodel.utils.ttnn_runtime import close_mesh

    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    prompts = args.prompt or ["What is the capital of France?"]
    ids = [tok(tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False),
               add_special_tokens=False).input_ids for p in prompts]
    mesh = _open(args)
    try:
        config, decoder = _decoder(args, mesh)
        eos = getattr(config, "eos_token_id", None)
        eos = eos if isinstance(eos, (list, tuple)) else [eos] if eos is not None else []
        t0 = time.time()
        out = decoder.generate(ids, args.max_new_tokens, eos_token_ids=eos)
        dt = time.time() - t0
        for prompt, gen in zip(prompts, out):
            print(f"\n>>> {prompt}\n{tok.decode(gen, skip_special_tokens=True)}", flush=True)
        steps = max(len(i) for i in ids) + max(len(g) for g in out) - 1
        print(f"\n[TT] {steps} lock-step positions in {dt:.2f}s ({steps / dt:.0f} steps/s)", flush=True)
    finally:
        close_mesh(owner="tt_llama_decode")


def cmd_bench(args):
    if args.cards > 1 and os.environ.get("_TT_LLAMA_CHILD") is None:
        return _bench_cards(args)
    import torch

    from gptqmodel.utils.ttnn_runtime import close_mesh

    mesh = _open(args)
    try:
        _config, decoder = _decoder(args, mesh)
        users = decoder.users
        tokens = torch.randint(100, 20000, (users,))
        positions = [args.pos] * users
        device_inputs = decoder.inputs(tokens, positions)
        out = decoder.capture(device_inputs)
        import ttnn

        ttnn.synchronize_device(mesh.device)
        t0 = time.perf_counter()
        for _ in range(args.iters):
            decoder.replay()
        ttnn.synchronize_device(mesh.device)
        traced = (time.perf_counter() - t0) / args.iters
        host = decoder.host_inputs(tokens, positions)
        t0 = time.perf_counter()
        for _ in range(args.iters):
            decoder.read_ids(decoder.replay(host))
        served = (time.perf_counter() - t0) / args.iters
        decoder.release()
        del out
        result = {"card": os.environ.get("TT_VISIBLE_DEVICES", "0"), "users": users, "pos": args.pos,
                  "traced_ms": traced * 1e3, "served_ms": served * 1e3,
                  "traced_tok_s": users / traced, "served_tok_s": users / served}
        print("[BENCH] " + json.dumps(result), flush=True)
    finally:
        close_mesh(owner="tt_llama_decode")


def _bench_cards(args):
    descriptor = _single_chip_descriptor()
    if descriptor is None:
        raise SystemExit("could not find the single-chip mesh graph descriptor in pjrt_plugin_tt")
    children = []
    for card in range(args.cards):
        env = dict(os.environ, TT_VISIBLE_DEVICES=str(card), TT_MESH_GRAPH_DESC_PATH=descriptor,
                   _TT_LLAMA_CHILD="1")
        children.append(subprocess.Popen([sys.executable, __file__, *sys.argv[1:]], env=env,
                                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
    results = []
    for card, child in enumerate(children):
        output, _ = child.communicate()
        lines = [l for l in output.splitlines() if l.startswith("[BENCH] ")]
        if child.returncode or not lines:
            print(output[-2000:], flush=True)
            raise SystemExit(f"card {card} failed (rc={child.returncode})")
        results.append(json.loads(lines[-1][len("[BENCH] "):]))
    for r in results:
        print(f"[BENCH] card {r['card']}: {r['users']} users @pos {r['pos']}: traced {r['traced_ms']:.2f} ms/step "
              f"({r['traced_tok_s']:.0f} tok/s), served {r['served_ms']:.2f} ms ({r['served_tok_s']:.0f} tok/s)")
    total_t = sum(r["traced_tok_s"] for r in results)
    total_s = sum(r["served_tok_s"] for r in results)
    print(f"[BENCH] {args.cards} cards x {results[0]['users']} users: {total_t:.0f} tok/s traced, "
          f"{total_s:.0f} tok/s served (host loop with readback)", flush=True)


def cmd_parity(args):
    import torch
    from transformers import AutoTokenizer

    from gptqmodel.utils.tt_blockfloat_quant import build_blockfloat_model
    from gptqmodel.utils.ttnn_runtime import close_mesh

    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    text = open(args.text).read() if args.text else (
        "The Tenstorrent Blackhole processor arranges Tensix cores in a grid connected by a network on chip. "
        "Each core has five RISC-V processors and a matrix engine. " * 20)
    ids = tok(text).input_ids
    seqs = torch.tensor([ids[i * 7: i * 7 + args.seq] for i in range(32)])
    reference, _, _ = build_blockfloat_model(args.checkpoint, dtype=torch.float32)
    with torch.no_grad():
        ref = reference(seqs).logits.float()
    del reference
    mesh = _open(args)
    try:
        _config, decoder = _decoder(args, mesh)
        got = []
        for pos in range(args.seq):
            live = decoder.inputs(seqs[:, pos], [pos] * decoder.users)
            got.append(decoder.read_logits(decoder.step(*live)))
        got = torch.stack(got, 1)
    finally:
        close_mesh(owner="tt_llama_decode")
    top1 = float((got.argmax(-1) == ref.argmax(-1)).float().mean())
    cos = float(torch.nn.functional.cosine_similarity(got, ref, dim=-1).mean())
    print(f"[PARITY] 32 x {args.seq} positions: top-1 agreement {top1:.4f}, mean logit cosine {cos:.5f}", flush=True)
    if top1 < args.min_top1:
        raise SystemExit(f"top-1 agreement {top1:.4f} below {args.min_top1}")


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("generate", "bench", "parity"):
        p = sub.add_parser(name)
        p.add_argument("checkpoint")
        p.add_argument("--max-seq", type=int, default=2048)
        p.add_argument("--fidelity", default="HiFi4", choices=("LoFi", "HiFi2", "HiFi3", "HiFi4"))
        p.add_argument("--kv-dtype", default="bfloat16", choices=("bfloat16", "bfloat8_b"))
        p.add_argument("--skip-drift", action="store_true", help="skip the per-tensor upload readback")
        if name == "generate":
            p.add_argument("--prompt", action="append")
            p.add_argument("--max-new-tokens", type=int, default=128)
        if name == "bench":
            p.add_argument("--pos", type=int, default=0)
            p.add_argument("--iters", type=int, default=50)
            p.add_argument("--cards", type=int, default=1)
        if name == "parity":
            p.add_argument("--text", default=None)
            p.add_argument("--seq", type=int, default=64)
            p.add_argument("--min-top1", type=float, default=0.95)
    args = parser.parse_args(argv)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.environ.setdefault("USE_TORCH_XLA", "0")
    {"generate": cmd_generate, "bench": cmd_bench, "parity": cmd_parity}[args.cmd](args)


if __name__ == "__main__":
    main(sys.argv[1:])
