import torch

from gptqmodel.models.definitions.baichuan import BaiChuanQModel


class _DummyRotary(torch.nn.Module):
    def __init__(self, inv_freq, *, max_seq_len_cached=32, base=10000):
        super().__init__()
        self.inv_freq = inv_freq
        self.max_seq_len_cached = max_seq_len_cached
        self.base = base


class _DummyAttention(torch.nn.Module):
    def __init__(self, rotary):
        super().__init__()
        self.rotary_emb = rotary


class _DummyLayer(torch.nn.Module):
    def __init__(self, rotary):
        super().__init__()
        self.self_attn = _DummyAttention(rotary)


class _DummyModel(torch.nn.Module):
    def __init__(self, rotary):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([_DummyLayer(rotary)])


def _new_qmodel():
    return object.__new__(BaiChuanQModel)


def test_after_model_load_materializes_meta_rotary_and_registers_buffers():
    rotary = _DummyRotary(torch.empty(8, device="meta"))
    model = _DummyModel(rotary)

    qmodel = _new_qmodel()
    returned = BaiChuanQModel.after_model_load(qmodel, model, load_quantized_model=False)

    assert returned is model
    assert rotary.inv_freq.device.type == "cpu"
    assert rotary.inv_freq.dtype == torch.float32
    assert rotary.cos_cached.shape == (1, 1, 32, 16)
    assert rotary.sin_cached.shape == (1, 1, 32, 16)
    assert set(rotary._buffers) == {"inv_freq", "cos_cached", "sin_cached"}
    assert rotary._non_persistent_buffers_set == {"inv_freq", "cos_cached", "sin_cached"}
    assert "inv_freq" not in rotary.__dict__
    assert "cos_cached" not in rotary.__dict__
    assert "sin_cached" not in rotary.__dict__


def test_after_model_load_promotes_existing_rotary_attrs_to_buffers():
    rotary = _DummyRotary(torch.arange(0, 8, dtype=torch.float32))
    rotary.cos_cached = torch.zeros((1, 1, 32, 16), dtype=torch.float32)
    rotary.sin_cached = torch.ones((1, 1, 32, 16), dtype=torch.float32)
    model = _DummyModel(rotary)

    qmodel = _new_qmodel()
    BaiChuanQModel.after_model_load(qmodel, model, load_quantized_model=False)

    assert set(rotary._buffers) == {"inv_freq", "cos_cached", "sin_cached"}
    assert rotary.cos_cached.device.type == "cpu"
    assert rotary.sin_cached.device.type == "cpu"
    assert "inv_freq" not in rotary.__dict__
    assert "cos_cached" not in rotary.__dict__
    assert "sin_cached" not in rotary.__dict__
