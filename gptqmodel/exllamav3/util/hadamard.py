from __future__ import annotations

import glob
import math
import os
from functools import lru_cache

import torch

from ..ext import exllamav3_ext as ext

had_dict: dict[int, torch.Tensor] = {}
primes: set[int] = set()
prime_limit = 1

def load_constants():
    global had_dict

    module_dir = os.path.dirname(os.path.abspath(__file__))
    had_dir = os.path.join(module_dir, "hadamard_data")
    file_pattern = os.path.join(had_dir, "hadamard_*.txt")
    files = glob.glob(file_pattern)
    had_dict = {}

    for file_path in files:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            lines = [line.strip() for line in lines if line.strip()]
            dim = len(lines)
            assert all(len(line) == dim for line in lines), "Non-square matrix in " + file_path
            matrix = [[1 if char == '+' else -1 for char in line] for line in lines]
            tensor = torch.tensor(matrix, dtype = torch.float16)
            had_dict[dim] = tensor

def ensure_primes(limit: int):
    global prime_limit, primes

    if limit <= prime_limit:
        return

    # Rebuild the cache when the requested Hadamard order needs a larger prime range.
    sieve = bytearray(b"\x01") * (limit + 1)
    sieve[:2] = b"\x00\x00"
    max_factor = math.isqrt(limit)
    for candidate in range(2, max_factor + 1):
        if not sieve[candidate]:
            continue
        start = candidate * candidate
        sieve[start:limit + 1:candidate] = b"\x00" * (((limit - start) // candidate) + 1)

    primes = {value for value, is_prime in enumerate(sieve) if is_prime}
    prime_limit = limit

def sylvester(h: torch.Tensor):
    d = h.shape[0]
    assert d == h.shape[1], "h not square"
    s = torch.empty((d * 2, d * 2), dtype = h.dtype, device = h.device)
    s[:d, :d] = h
    s[:d, d:] = h
    s[d:, :d] = h
    s[d:, d:] = -h
    return s

def is_quadratic_residue(a: int, p: int):
    return pow(a, (p - 1) // 2, p) == 1

def paley_torch(n: int):
    h = torch.empty((n, n), dtype = torch.half)
    p = n - 1
    for i in range(p):
        for j in range(p):
            if i == j:
                h[i + 1][j + 1] = 1
            else:
                residue = (i - j) % p
                if is_quadratic_residue(residue, p):
                    h[i + 1][j + 1] = 1
                else:
                    h[i + 1][j + 1] = -1
    h[0, :] = 1
    h[:, 0] = -1
    h[0, 0] = 1
    return h

def paley(n: int):
    h = torch.empty((n, n), dtype = torch.half)
    ext.had_paley(h)
    # ref = paley_torch(n)
    # assert torch.all(h == ref)
    return h

def paley2_torch(n: int):
    h = torch.empty((n, n), dtype = torch.half)
    p = n // 2 - 1
    for i in range(n // 2):
        i0 = 2 * i + 0
        i1 = 2 * i + 1
        for j in range(n // 2):
            j0 = 2 * j + 0
            j1 = 2 * j + 1
            if j == i:
                h[i0, j0] = 1
                h[i0, j1] = -1
                h[i1, j0] = -1
                h[i1, j1] = -1
            else:
                residue = (i - j) % p
                if i == 0 or j == 0 or is_quadratic_residue(residue, p):
                    h[i0, j0] = 1
                    h[i0, j1] = 1
                    h[i1, j0] = 1
                    h[i1, j1] = -1
                else:
                    h[i0, j0] = -1
                    h[i0, j1] = -1
                    h[i1, j0] = -1
                    h[i1, j1] = 1
    return h

def paley2(n: int):
    h = torch.empty((n, n), dtype = torch.half)
    ext.had_paley2(h)
    # ref = paley2_torch(n)
    # assert torch.all(h == ref)
    return h

@lru_cache(maxsize = 100)
def get_hadamard(n: int):
    global had_dict

    if not had_dict:
        load_constants()

    if n in had_dict: return had_dict[n]

    # Sylvester's construction
    if n % 2 == 0:
        s = get_hadamard(n // 2)
        if s is not None:
            s = sylvester(s)
            return s

    if n % 4 == 0:
        ensure_primes(max(n - 1, (n // 2) - 1))

    # Paley construction
    if n % 4 == 0 and (n - 1) % 4 == 3 and (n - 1) in primes:
        return paley(n)

    # Other Paley construction
    if n % 4 == 0 and (n // 2) - 1 in primes:
        return paley2(n)

    return None

@lru_cache(maxsize = 100)
def get_hadamard_dt(n: int, device: torch.device | str, dtype: torch.dtype, scale = 1.0):
    had = get_hadamard(n).to(device = device, dtype = dtype, copy = True)
    had *= scale
    return had
