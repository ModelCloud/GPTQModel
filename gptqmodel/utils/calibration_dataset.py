def batched(iterable, n: int, format_func):
    # batched('ABCDEFG', 3) → ABC DEF G
    assert n >= 1, "batch size must be at least one"
    from itertools import islice

    iterator = iter(iterable)

    while batch := tuple(islice(iterator, n)):
        if format_func is None:
            yield batch
        else:
            formatted_batch = [format_func(item) for item in batch]
            yield formatted_batch
