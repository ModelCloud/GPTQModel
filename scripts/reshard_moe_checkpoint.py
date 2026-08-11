#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Create a module-tree-driven, per-layer MoE checkpoint layout."""

import argparse

from gptqmodel import ShardStrategy, reshard


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--moe-modules-per-shard", type=int, default=128)
    parser.add_argument("--num-write-workers", type=int, default=8)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()
    result = reshard(
        args.source,
        args.output,
        strategy=ShardStrategy.PER_LAYER_MOE,
        moe_modules_per_shard=args.moe_modules_per_shard,
        num_write_workers=args.num_write_workers,
        trust_remote_code=args.trust_remote_code,
    )
    print(
        f"created {result['num_shards']} shards, {result['total_bytes'] / 1024**3:.2f} GiB, "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
