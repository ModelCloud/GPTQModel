import gc
import weakref

from gptqmodel.utils import inspect as inspect_utils
from gptqmodel.utils.inspect import get_supported_kwargs, safe_kwargs_call


class _CallableWithoutVarKwargs:
    def __call__(self, hidden_states, attention_mask=None, use_cache=False):
        return hidden_states, attention_mask, use_cache


def _clear_supported_kwargs_cache():
    inspect_utils._get_supported_kwargs_cached.cache_clear()


def test_get_supported_kwargs_does_not_keep_bound_callable_alive():
    _clear_supported_kwargs_cache()
    callable_obj = _CallableWithoutVarKwargs()
    obj_ref = weakref.ref(callable_obj)

    # Bound Python methods should be normalized to the shared function object,
    # so this lookup must not keep the instance alive.
    accepts_var_kw, allowed_kwargs = get_supported_kwargs(callable_obj.__call__)

    assert accepts_var_kw is False
    assert allowed_kwargs is not None
    assert "attention_mask" in allowed_kwargs
    assert "use_cache" in allowed_kwargs

    del callable_obj
    gc.collect()

    assert obj_ref() is None


def test_get_supported_kwargs_caches_bound_methods_by_unbound_function(monkeypatch):
    _clear_supported_kwargs_cache()

    signature_calls = []
    original_signature = inspect_utils.inspect.signature

    def counting_signature(callable_obj):
        signature_calls.append(callable_obj)
        return original_signature(callable_obj)

    monkeypatch.setattr(inspect_utils.inspect, "signature", counting_signature)

    first = _CallableWithoutVarKwargs()
    second = _CallableWithoutVarKwargs()

    # Two bound methods from different instances should collapse onto the same
    # underlying function cache key.
    get_supported_kwargs(first.__call__)
    get_supported_kwargs(second.__call__)

    assert signature_calls == [_CallableWithoutVarKwargs.__call__]


def test_get_supported_kwargs_caches_module_level_builtins(monkeypatch):
    _clear_supported_kwargs_cache()

    signature_calls = []
    original_signature = inspect_utils.inspect.signature

    def counting_signature(callable_obj):
        signature_calls.append(callable_obj)
        return original_signature(callable_obj)

    monkeypatch.setattr(inspect_utils.inspect, "signature", counting_signature)

    # Module-level builtins such as len are process-stable and safe to cache.
    get_supported_kwargs(len)
    get_supported_kwargs(len)

    assert signature_calls == [len]


def test_get_supported_kwargs_does_not_cache_instance_bound_builtin_methods(monkeypatch):
    _clear_supported_kwargs_cache()

    signature_calls = []
    original_signature = inspect_utils.inspect.signature

    def counting_signature(callable_obj):
        signature_calls.append(callable_obj)
        return original_signature(callable_obj)

    monkeypatch.setattr(inspect_utils.inspect, "signature", counting_signature)

    values = []
    append = values.append

    # Instance-bound builtin methods retain their owner through __self__, so
    # they must stay on the uncached path.
    get_supported_kwargs(append)
    get_supported_kwargs(append)

    assert len(signature_calls) == 2


def test_safe_kwargs_call_filters_unsupported_kwargs():
    _clear_supported_kwargs_cache()
    callable_obj = _CallableWithoutVarKwargs()

    # safe_kwargs_call should preserve supported kwargs while dropping extras.
    result = safe_kwargs_call(
        callable_obj,
        "hidden",
        kwargs={
            "attention_mask": "mask",
            "use_cache": True,
            "unsupported": "drop-me",
        },
    )

    assert result == ("hidden", "mask", True)
