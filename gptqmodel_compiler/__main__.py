# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""python -m gptqmodel_compiler --program graph.json --target eager --output out"""

import argparse
import json
from pathlib import Path

from .compiler import compile_program
from .ir import Program
from .llama import llama32_1b


def main():
    parser = argparse.ArgumentParser(description="Compile typed DSL to standalone PyTorch 2.15 code")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--program", type=Path, help="serialized Python DSL program")
    source.add_argument("--llama32-1b", action="store_true", help="instantiate the first Llama DSL sample")
    parser.add_argument("--dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    parser.add_argument("--capacity", type=int, default=131072)
    parser.add_argument("--target", choices=["eager", "optimized"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    program = (
        Program.from_dict(json.loads(args.program.read_text()))
        if args.program
        else llama32_1b(dtype=args.dtype, capacity=args.capacity)
    )
    artifact = compile_program(program, target=args.target)
    print(artifact.write(args.output))


if __name__ == "__main__":
    main()
