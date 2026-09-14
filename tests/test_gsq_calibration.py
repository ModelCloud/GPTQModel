import pytest
import torch

from gptqmodel.quantization.gsq_calibration import GSQInputGram


def test_weighted_masked_batches_match_explicit_activations():
    rng = torch.Generator().manual_seed(7)
    x = torch.randn(2, 5, 8, generator=rng)
    y = torch.randn(3, 8, generator=rng)
    mask = torch.tensor([[True, True, False, False, True], [True, False, True, True, False]])
    capture = GSQInputGram(8)
    capture.add(x, mask=mask, source_weight=1.25)
    capture.add(y)
    expected = 1.25 * (x[mask].T @ x[mask]) + y.T @ y
    # Capture must not alias mutable caller inputs.
    x.zero_()
    gram, stats = capture.take()
    torch.testing.assert_close(gram, expected)
    assert stats == {'tokens': 9, 'weighted_tokens': 10.5}
    assert capture._gram is None
    with pytest.raises(RuntimeError, match='closed'):
        capture.add(y)
    with pytest.raises(RuntimeError, match='closed'):
        capture.take()


def test_failed_batch_does_not_corrupt_prior_statistics():
    capture = GSQInputGram(4)
    capture.add(torch.ones(2, 4))
    with pytest.raises(ValueError, match='overflow'):
        capture.add(torch.full((2, 4), 1e30))
    gram, stats = capture.take()
    assert torch.equal(gram, torch.full((4, 4), 2.))
    assert stats['tokens'] == 2


def test_zero_weight_and_empty_mask_do_not_fake_calibration():
    capture = GSQInputGram(4)
    capture.add(torch.ones(2, 4), source_weight=0)
    capture.add(torch.full((2, 4), float('nan')), mask=torch.zeros(2, dtype=torch.bool))
    with pytest.raises(ValueError, match='no positive-weight'):
        capture.take()
    capture.clear()
    with pytest.raises(RuntimeError, match='closed'):
        capture.take()


@pytest.mark.parametrize('weight', [-1, float('inf'), True])
def test_invalid_source_weight_rejected(weight):
    with pytest.raises(ValueError, match='source weight'):
        GSQInputGram(4).add(torch.ones(2, 4), source_weight=weight)


def test_memory_guard_precedes_allocation():
    with pytest.raises(ValueError, match='memory budget'):
        GSQInputGram(4096, max_bytes=16)


def test_concurrent_capture_counts_every_batch_once():
    from concurrent.futures import ThreadPoolExecutor

    capture = GSQInputGram(4)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _: capture.add(torch.ones(2, 4)), range(16)))
    gram, stats = capture.take()
    assert torch.equal(gram, torch.full((4, 4), 32.))
    assert stats['tokens'] == stats['weighted_tokens'] == 32
