# GPU=-1
import random
import string

import pytest

from gptqmodel.utils.random_str import get_random_string


def test_default_length():
    s = get_random_string()
    assert len(s) == 8


def test_custom_length():
    s = get_random_string(16)
    assert len(s) == 16


def test_characters_are_lowercase_letters():
    s = get_random_string(100)
    assert set(s).issubset(set(string.ascii_lowercase))


def test_multiple_calls_produce_different_values():
    results = [get_random_string() for _ in range(5)]
    assert len(set(results)) == 5


def test_not_affected_by_random_seed():
    random.seed(42)
    r1 = get_random_string()

    random.seed(42)
    r2 = get_random_string()

    # Not affected by the seed → Should not be exactly identical
    assert r1 != r2, f"{r1} and {r2} Should not be exactly identical"


def test_length_zero():
    s = get_random_string(0)
    assert s == ""


@pytest.mark.parametrize("length", [1, 2, 10, 50])
def test_various_lengths(length):
    s = get_random_string(length)
    assert len(s) == length
