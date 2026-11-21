# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Standalone model DSL compiler. Importing this package never imports Transformers."""

from .compiler import compile_program
from .ir import Block, Operation, Parameter, Program, Resource, ResourceUse, Tensor
from .llama import llama32_1b, llama_program


__all__ = [
    "Block",
    "Operation",
    "Parameter",
    "Program",
    "Resource",
    "ResourceUse",
    "Tensor",
    "compile_program",
    "llama32_1b",
    "llama_program",
]
