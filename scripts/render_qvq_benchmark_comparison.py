"""Render a Markdown comparison between two QVQ LR benchmark JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _key(row: dict) -> tuple:
    return (
        row.get("name", row.get("shape")),
        row["m"],
        row["k"],
        row["n"],
        float(row["bits"]),
    )


def render(current: dict, previous: dict) -> str:
    previous_candidates = [
        row
        for row in previous["rows"]
        if row.get("state") == "complete" and row.get("kernel") == "qvq_lr"
    ]
    if not previous_candidates:
        # Standalone optimization experiments label the candidate instead of
        # the production kernel. Exclude their same-run control rows.
        previous_candidates = [
            row for row in previous["rows"] if row.get("candidate") not in (None, "last_packed")
        ]
    previous_rows = {_key(row): row for row in previous_candidates}
    rows = [
        row
        for row in current["rows"]
        if row.get("state") == "complete" and row.get("kernel") == "qvq_lr"
    ]
    rows.sort(key=lambda row: (row["name"], row["m"], float(row["bits"])))

    device = current["device"]
    hardware = current["hardware"]
    args = current["args"]
    lines = [
        "# QVQ LR H100 comparison — focused M sweep",
        "",
        (
            "QVQ V2B2-P32-LR versus matched W4 GPTQ Marlin/Machete baselines "
            "for the Llama 3.2 1B projection shapes."
        ),
        "",
        "## Measurement contract",
        "",
        f"- Current benchmark commit: `{current['commit']}`",
        f"- Previous benchmark commit: `{previous['commit']}`",
        (
            f"- GPU: physical `{current['physical_gpu']}`, `{device['name']}`, "
            f"PCI `{hardware['pci.bus_id']}`, UUID `{hardware['uuid']}`, "
            f"CC `{device['compute_capability']}`, {device['sm_count']} SMs"
        ),
        f"- M values: `{args['m']}`; QVQ rates: `{current['qvq']['rates']}`",
        "- Timing: one CUDA Graph replay with internal external CUDA events; host launch gaps excluded.",
        "- `xMarlin`/`xMachete` = baseline median latency / QVQ median latency; above 1.0x is faster.",
        "- `Better than last` is `yes` only when current QVQ median latency is strictly lower than the previous run; equal/regressed is `no`.",
        "",
        "| Shape | Roles | M | K | N | W | Median ms | Logical TFLOP/s | xMarlin | xMachete | Better than last |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        previous_row = previous_rows.get(_key(row))
        better = previous_row is not None and row["median_ms"] < previous_row["median_ms"]
        lines.append(
            f"| {row['name']} | {'/'.join(row['roles'])} | {row['m']} | {row['k']} | {row['n']} | "
            f"{float(row['bits']):g} | {row['median_ms']:.4f} | {row['logical_tflops']:.3f} | "
            f"{row['speedup_vs_marlin']:.3f}x | {row['speedup_vs_machete']:.3f}x | "
            f"{'yes' if better else 'no'} |"
        )
    lines.extend([
        "",
        (
            "The four unique geometries preserve all seven Llama 3.2 1B roles: "
            "q_proj/o_proj (2048×2048), k_proj/v_proj (2048×512), "
            "gate_proj/up_proj (2048×8192), and down_proj (8192×2048)."
        ),
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("current", type=Path)
    parser.add_argument("previous", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    current = json.loads(args.current.read_text(encoding="utf-8"))
    previous = json.loads(args.previous.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(current, previous), encoding="utf-8")


if __name__ == "__main__":
    main()
