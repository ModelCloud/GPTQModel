# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Serve, benchmark or check an out-axis block-float Llama on Tenstorrent.

Usage::

    # chat generation on one card
    python scripts/tt_llama_decode.py generate CKPT --prompt "What is 17 * 23?"
    # decode step and prefill throughput at a context length, one process per card
    python scripts/tt_llama_decode.py bench CKPT --ctx 16384 --users 32 --kv-dtype bfloat8_b --cards 4
    # long-context passkey retrieval
    python scripts/tt_llama_decode.py passkey CKPT --ctx 32768

``CKPT`` must be a tt-blockfloat checkpoint written with ``block_axis="out"``
(``scripts/tt_quantize_blockfloat.py --block-axis out``).

Multi-card throughput is data parallel with one process per card: each child
sees a single chip (``TT_VISIBLE_DEVICES``, the Tenstorrent runtime's own
selector) and opens it with the single-chip fabric descriptor.  That measured
linear scaling where a 1xN ``MeshDevice`` running the same trace was ~2x slower
per step.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import subprocess
import sys
import time


TRACE_REGION = 200_000_000
BUCKETS = (1, 2, 4, 8, 16, 32)


def _single_chip_descriptor():
    try:
        import pjrt_plugin_tt
    except Exception:
        return None
    root = os.path.join(os.path.dirname(pjrt_plugin_tt.__file__), "tt-metal")
    hits = glob.glob(os.path.join(root, "tt_metal", "fabric", "mesh_graph_descriptors",
                                  "p150_mesh_graph_descriptor.textproto"))
    return hits[0] if hits else None


def _model(args, *, max_context, kv_tokens):
    from gptqmodel.utils.tt_llama import TTLlama, load_tt_llama_weights
    from gptqmodel.utils.ttnn_runtime import open_mesh

    config, weights = load_tt_llama_weights(args.checkpoint)
    mesh = open_mesh(device_count=1, trace_region_size=TRACE_REGION, owner="tt_llama_decode")
    t0 = time.time()
    model = TTLlama(mesh.device, config, weights, max_context=max_context, kv_tokens=kv_tokens,
                    fidelity=args.fidelity, kv_dtype=args.kv_dtype, check_drift=not args.skip_drift)
    del weights
    drift = max(model.drift.values()) if model.drift else float("nan")
    print(f"[TT] uploaded in {time.time() - t0:.1f}s; {len(model.drift)} block-float tensors, max drift "
          f"{drift:.1e}; KV pool {model.kv_bytes() / 2**30:.2f} GiB", flush=True)
    if model.drift and drift != 0.0:
        raise SystemExit("[TT] ERROR: block-float upload is not exact; is this an out-axis checkpoint?")
    return mesh, model


def _close():
    from gptqmodel.utils.ttnn_runtime import close_mesh

    close_mesh(owner="tt_llama_decode")


def _eos(config):
    eos = getattr(config, "eos_token_id", None)
    return eos if isinstance(eos, (list, tuple)) else [eos] if eos is not None else []


def cmd_generate(args):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    prompts = args.prompt or ["What is the capital of France?"]
    ids = [tok(tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False),
               add_special_tokens=False).input_ids for p in prompts]
    longest = max(len(i) for i in ids) + args.max_new_tokens
    _mesh, model = _model(args, max_context=longest, kv_tokens=len(ids) * longest + 256)
    try:
        t0 = time.time()
        out = model.generate(ids, args.max_new_tokens, eos_token_ids=_eos(model.config))
        dt = time.time() - t0
        for prompt, gen in zip(prompts, out):
            print(f"\n>>> {prompt}\n{tok.decode(gen, skip_special_tokens=True)}", flush=True)
        print(f"\n[TT] {sum(len(g) for g in out)} tokens for {len(out)} prompts in {dt:.2f}s", flush=True)
    finally:
        _close()


def cmd_bench(args):
    if args.cards > 1 and args.card is None:
        return _bench_cards(args)
    import torch
    import ttnn

    if args.host_threads:
        # Four card processes each defaulting to every core oversubscribe the
        # host and turn the per-step gather/upload into the bottleneck.
        torch.set_num_threads(args.host_threads)
    users = args.users
    ctx = args.ctx
    mesh, model = _model(args, max_context=ctx + 128, kv_tokens=users * (ctx + 128) + 32 * 128)
    try:
        # Prefill throughput on a real-length prompt (warm), user 0.
        prompt = torch.randint(100, 20000, (ctx,)).tolist()
        model.prefill(0, prompt[: min(ctx, 2 * model.prefill_chunk)])  # compile
        t0 = time.time()
        model.prefill(0, prompt)
        prefill_s = time.time() - t0
        # Decode: `users` slots at position ctx (timing only; KV beyond the prompt is not meaningful).
        for user in range(users):
            model.pool.reserve(user, ctx + 64)
        model._sync_page_table()
        active = next(b for b in BUCKETS if b >= users)
        positions = [ctx] * users + [0] * (32 - users)
        model.capture([1] * 32, positions, active)
        ttnn.synchronize_device(mesh.device)
        t0 = time.perf_counter()
        for _ in range(args.iters):
            model.replay()
        ttnn.synchronize_device(mesh.device)
        step = (time.perf_counter() - t0) / args.iters
        t0 = time.perf_counter()
        for _ in range(args.iters):
            model.read_ids(model.replay([1] * 32, positions))
        served = (time.perf_counter() - t0) / args.iters
        model.release()
        result = {"card": args.card or 0, "ctx": ctx, "users": users, "kv": args.kv_dtype,
                  "prefill_tok_s": round(ctx / prefill_s), "step_ms": round(step * 1e3, 2),
                  "served_step_ms": round(served * 1e3, 2), "tok_s": round(users / step),
                  "served_tok_s": round(users / served), "per_user_tok_s": round(1 / step, 1)}
        print("[BENCH] " + json.dumps(result), flush=True)
    finally:
        _close()


def _bench_cards(args):
    descriptor = _single_chip_descriptor()
    if descriptor is None:
        raise SystemExit("could not find the single-chip mesh graph descriptor in pjrt_plugin_tt")
    children = []
    for card in range(args.cards):
        env = dict(os.environ, TT_VISIBLE_DEVICES=str(card), TT_MESH_GRAPH_DESC_PATH=descriptor)
        threads = str(max(1, (os.cpu_count() or 1) // args.cards))
        children.append(subprocess.Popen([sys.executable, __file__, *sys.argv[1:], "--card", str(card),
                                          "--host-threads", threads], env=env,
                                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
    results = []
    for card, child in enumerate(children):
        output, _ = child.communicate()
        lines = [line for line in output.splitlines() if line.startswith("[BENCH] ")]
        if child.returncode or not lines:
            print(output[-2000:], flush=True)
            raise SystemExit(f"card {card} failed (rc={child.returncode})")
        results.append(json.loads(lines[-1][len("[BENCH] "):]))
    for r in results:
        print("[BENCH] " + json.dumps(r), flush=True)
    print(f"[BENCH] {args.cards} cards x {args.users} users @ctx {args.ctx}: "
          f"{sum(r['tok_s'] for r in results)} tok/s traced, {sum(r['served_tok_s'] for r in results)} tok/s served",
          flush=True)


def _passkey_prompt(tok, ctx, key, depth, filler):
    template = tok.apply_chat_template([{"role": "user", "content": "PLACEHOLDER"}], add_generation_prompt=True,
                                       tokenize=False)
    pre, post = (tok(t, add_special_tokens=False).input_ids for t in template.split("PLACEHOLDER"))
    ask = tok("\n\nWhat is the secret passkey mentioned in the text above? Answer with the number only.",
              add_special_tokens=False).input_ids
    needle = tok(f"\n\nThe secret passkey is {key}. Remember it: {key}.\n\n", add_special_tokens=False).input_ids
    budget = ctx - len(pre) - len(ask) - len(post) - len(needle) - 64
    body = (filler * (budget // len(filler) + 1))[:budget]
    cut = int(budget * depth)
    return pre + body[:cut] + needle + body[cut:] + ask + post


def cmd_passkey(args):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    if args.filler:
        with open(args.filler, encoding="utf-8") as f:
            filler_text = f.read()
        filler = tok(filler_text, add_special_tokens=False).input_ids
    else:
        filler = tok("The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. ",
                     add_special_tokens=False).input_ids
    _mesh, model = _model(args, max_context=args.ctx + 64, kv_tokens=args.ctx + 32 * 128)
    try:
        hits = 0
        for depth in args.depth:
            key = random.Random(args.ctx + int(depth * 100)).randint(10000, 99999)
            prompt = _passkey_prompt(tok, args.ctx, key, depth, filler)
            t0 = time.time()
            out = model.generate([prompt], 8, eos_token_ids=_eos(model.config))[0]
            text = tok.decode(out, skip_special_tokens=True).strip()
            ok = str(key) in text
            hits += ok
            print(f"[PASSKEY] ctx {len(prompt)} depth {depth}: key {key} answer {text[:24]!r} "
                  f"{'OK' if ok else 'MISS'} ({time.time() - t0:.1f}s)", flush=True)
        print(f"[PASSKEY] {hits}/{len(args.depth)} retrieved", flush=True)
    finally:
        _close()


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("generate", "bench", "passkey"):
        p = sub.add_parser(name)
        p.add_argument("checkpoint")
        p.add_argument("--fidelity", default="HiFi4", choices=("LoFi", "HiFi2", "HiFi3", "HiFi4"))
        p.add_argument("--kv-dtype", default="bfloat16", choices=("bfloat16", "bfloat8_b"))
        p.add_argument("--skip-drift", action="store_true", help="skip the per-tensor upload readback")
        if name == "generate":
            p.add_argument("--prompt", action="append")
            p.add_argument("--max-new-tokens", type=int, default=128)
        if name == "bench":
            p.add_argument("--ctx", type=int, default=2048)
            p.add_argument("--users", type=int, default=32, choices=range(1, 33), metavar="1..32")
            p.add_argument("--iters", type=int, default=30)
            p.add_argument("--cards", type=int, default=1)
            p.add_argument("--card", type=int, default=None, help=argparse.SUPPRESS)
            p.add_argument("--host-threads", type=int, default=None,
                           help="torch CPU threads for the host loop (per-card share when --cards > 1)")
        if name == "passkey":
            p.add_argument("--ctx", type=int, default=8192)
            p.add_argument("--depth", type=float, action="append")
            p.add_argument("--filler", default=None, help="text file used as the haystack")
    args = parser.parse_args(argv)
    if args.cmd == "passkey" and not args.depth:
        args.depth = [0.25, 0.5, 0.75]
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.environ.setdefault("USE_TORCH_XLA", "0")
    {"generate": cmd_generate, "bench": cmd_bench, "passkey": cmd_passkey}[args.cmd](args)


if __name__ == "__main__":
    main(sys.argv[1:])
