# SPDX-FileCopyrightText: 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from gptqmodel.looper.module_looper import ModuleLooper


@pytest.mark.parametrize("initial", (True, False, None))
@pytest.mark.parametrize("fails", (True, False))
def test_module_looper_restores_exact_use_cache_state_on_every_exit(monkeypatch, initial, fails):
    config = SimpleNamespace()
    if initial is not None:
        config.use_cache = initial
    looper = object.__new__(ModuleLooper)
    looper.gptq_model = SimpleNamespace(model=SimpleNamespace(config=config))

    def loop_impl(self, fallback=None, **kwargs):
        del self, fallback, kwargs
        assert config.use_cache is False
        config.use_cache = "mutated during quantization"
        if fails:
            raise RuntimeError("quantization failed")
        return {"ok": True}

    monkeypatch.setattr(ModuleLooper, "_loop_impl", loop_impl)
    if fails:
        with pytest.raises(RuntimeError, match="quantization failed"):
            looper.loop()
    else:
        assert looper.loop() == {"ok": True}

    if initial is None:
        assert not hasattr(config, "use_cache")
    else:
        assert config.use_cache is initial
