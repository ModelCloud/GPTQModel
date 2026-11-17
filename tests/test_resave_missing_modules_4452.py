import torch

import gptqmodel.utils.model as model_utils


class _ModelWithoutLegacyModule:
    def named_modules(self):
        return []

    def modules(self):
        return []

    def get_submodule(self, name):
        raise AttributeError(name)


def test_checkpoint_dtype_restore_skips_removed_module(monkeypatch, tmp_path):
    monkeypatch.setattr(model_utils, "_prepare_variable_quant_buffers", lambda *_args: None)
    monkeypatch.setattr(model_utils.accelerate, "load_checkpoint_in_model", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        model_utils,
        "_checkpoint_tensor_dtypes",
        lambda _checkpoint: {"model.layers.0.self_attn.rotary_emb.inv_freq": torch.float32},
    )
    monkeypatch.setattr(
        model_utils, "_tie_weights_after_checkpoint_load", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(model_utils, "release_pinned_host_cache", lambda: None)

    model_utils.load_checkpoint_in_model_then_tie_weights(
        _ModelWithoutLegacyModule(),
        checkpoint=str(tmp_path / "checkpoint"),
        restore_checkpoint_buffer_dtypes=True,
    )
