def batched(iterable, n: int, process_func):
    # batched('ABCDEFG', 3) → ABC DEF G
    assert n >= 1, "batch size must be at least one"
    from itertools import islice

    iterator = iter(iterable)

    while batch := tuple(islice(iterator, n)):
        if process_func is None:
            yield batch
        else:
            yield [process_func(item) for item in batch]
