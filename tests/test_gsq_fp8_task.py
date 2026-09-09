import pytest
import torch

from gptqmodel.nn_modules.qlinear.fp8 import TorchFP8Linear
from gptqmodel.quantization.config import FP8Config
from gptqmodel.quantization.gsq_fp8_task import FP8GSQTask


@pytest.mark.parametrize('method', ['row', 'tensor', 'block'])
def test_capture_fit_export_and_release(method):
    rng = torch.Generator().manual_seed(7)
    teacher = torch.randn(4, 8, generator=rng)
    train = torch.randn(2, 12, 8, generator=rng)
    mask = torch.ones(2, 12, dtype=torch.bool)
    mask[:, -2:] = False
    cfg = FP8Config(weight_scale_method=method, weight_block_size=[2, 4] if method == 'block' else None,
                    gsq={'enabled': True, 'steps': 5, 'candidates': 3})
    task = FP8GSQTask(teacher, cfg)
    task.add_batch(train, mask=mask, source_weight=1.25)
    result = task.quantize()
    assert task.teacher is None
    assert task.capture._gram is None
    packed = TorchFP8Linear(bits=8, group_size=-1, sym=True, desc_act=False, in_features=8, out_features=4,
                           bias=False, weight_scale_method=method, weight_block_size=cfg.weight_block_size)
    packed.weight.copy_(result['weight'])
    packed.weight_scale_inv.copy_(result['scale_inv'])
    x = train[mask]
    expected = float(((packed(x)-x @ teacher.T).square().sum())/((x @ teacher.T).square().sum()))
    assert result['after'] == pytest.approx(expected, rel=2e-4, abs=1e-8)
    assert result['after'] <= result['before']
    assert result['diagnostics']['tokens'] == 20
    assert result['diagnostics']['weighted_tokens'] == 25
    with pytest.raises(RuntimeError, match='consumed'):
        task.quantize()


def test_fitting_failure_releases_teacher_and_capture(monkeypatch):
    import gptqmodel.quantization.gsq_fp8_task as implementation

    task = FP8GSQTask(torch.ones(4, 8), FP8Config(gsq={'enabled': True}))
    task.add_batch(torch.ones(2, 8))

    def fail(*args, **kwargs):
        raise ValueError('injected fitting failure')

    monkeypatch.setattr(implementation, 'refine_fp8_weight', fail)
    with pytest.raises(ValueError, match='injected'):
        task.quantize()
    assert task.teacher is None
    assert task.capture._gram is None



def test_calibrated_task_uses_same_smoothed_initializer_as_packer(monkeypatch):
    import gptqmodel.quantization.gsq_fp8_task as implementation
    from gptqmodel.quantization.config import SmoothMAD

    rng = torch.Generator().manual_seed(7)
    teacher = torch.randn(4, 64, generator=rng)
    teacher[:, 0] = 100
    cfg = FP8Config(smooth=SmoothMAD(k=2.), gsq={'enabled': True})
    dense = torch.nn.Linear(64, 4, bias=False)
    dense.weight.data.copy_(teacher)
    baseline = TorchFP8Linear(bits=8, group_size=-1, sym=True, desc_act=False,
                              in_features=64, out_features=4, bias=False)
    baseline.pack_original(dense, None, None, smooth=cfg.smooth)
    task = FP8GSQTask(teacher, cfg)
    task.add_batch(torch.ones(2, 64))

    def verify(weight, scales, *, target, **kwargs):
        assert torch.equal(weight.view(torch.uint8), baseline.weight.view(torch.uint8))
        assert torch.equal(scales, baseline.weight_scale_inv)
        assert torch.equal(target, teacher)
        return {'weight': weight, 'scale_inv': scales, 'before': 1., 'after': 1., 'history': [1.]}

    monkeypatch.setattr(implementation, 'refine_fp8_weight', verify)
    task.quantize()
