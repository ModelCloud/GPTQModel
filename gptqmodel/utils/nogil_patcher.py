# SPDX-FileCopyrightText: 2024-2025 ModelCloud.ai
# SPDX-FileCopyrightText: 2024-2025 qubitium@modelcloud.ai
# SPDX-License-Identifier: Apache-2.0
# Contact: qubitium@modelcloud.ai, x.com/qubitium

"""Straightforward monkey patch helpers for nogil runtimes."""

import threading
import time

from .safe import ThreadSafe


_PATCHED_ATTR = "_gptqmodel_locked_save_file"


def patch_safetensors_save_file() -> None:
    try:
        from safetensors import torch as safetensors_torch
    except ImportError:
        return

    if getattr(safetensors_torch.save_file, _PATCHED_ATTR, False):
        return

    wrapper = ThreadSafe(safetensors_torch).save_file
    setattr(wrapper, _PATCHED_ATTR, True)
    safetensors_torch.save_file = wrapper


__all__ = ["patch_safetensors_save_file", "patch_triton_autotuner"]


def patch_triton_autotuner() -> None:
    try:
        import triton
        from triton.runtime import autotuner as module
    except ImportError:
        return

    version = getattr(triton, "__version__", None)
    if version is None or tuple(int(part) for part in version.split(".")[:3]) < (3, 5, 0):
        return

    autotuner_cls = module.Autotuner
    if getattr(autotuner_cls, "_gptqmodel_threadsafe", False):
        return

    builtins_mod = module.builtins
    Config = module.Config
    driver = module.driver
    knobs = module.knobs
    get_cache_manager = module.get_cache_manager
    triton_key = module.triton_key
    get_cache_invalidating_env_vars = module.get_cache_invalidating_env_vars
    JITFunction = module.JITFunction
    hashlib_mod = module.hashlib
    json_mod = module.json
    class CacheFuture:
        __slots__ = ("event", "config", "error", "used_cached_result", "bench_time")

        def __init__(self):
            self.event = threading.Event()
            self.config = None
            self.error = None
            self.used_cached_result = True
            self.bench_time = None

    module.CacheFuture = CacheFuture

    original_init = autotuner_cls.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        cache_map = getattr(self, "cache", {})
        self._cache = dict(cache_map)
        self.cache = self._cache
        self._cache_lock = getattr(self, "_cache_lock", threading.RLock())
        self._cache_futures = {}

    def patched_check_disk_cache(self, tuning_key, configs, bench_fn):
        if not tuning_key or any(cfg.pre_hook for cfg in configs):
            configs_timings, bench_time, best_config = bench_fn()
            self.configs_timings = configs_timings
            return False, bench_time, configs_timings, best_config

        from triton.compiler.compiler import make_backend

        fn = self.fn
        while not isinstance(fn, JITFunction):
            fn = fn.fn

        env_vars = get_cache_invalidating_env_vars()
        cache_key = [
            triton_key(),
            make_backend(driver.active.get_current_target()).hash(),
            fn.cache_key,
            str(sorted(env_vars.items())),
            str(tuning_key),
        ] + [str(c) for c in configs]
        cache_key = hashlib_mod.sha256("-".join(cache_key).encode("utf-8")).hexdigest()
        cache = get_cache_manager(cache_key)
        file_name = f"{fn.__name__[:150]}.autotune.json"
        path = cache.get_file(file_name)
        if path:
            with open(path, "r") as cached_configs:
                timings = json_mod.load(cached_configs)["configs_timings"]
                configs_timings = {Config(**config): timing for config, timing in timings}
            self.configs_timings = configs_timings
            best_config = builtins_mod.min(configs_timings, key=configs_timings.get)
            return True, None, configs_timings, best_config

        configs_timings, bench_time, best_config = bench_fn()
        self.configs_timings = configs_timings
        cache.put(
            json_mod.dumps({
                "key": tuning_key,
                "configs_timings": [
                    (config.__dict__, timings)
                    for config, timings in (configs_timings or {}).items()
                    if not config.pre_hook
                ],
            }),
            file_name,
            binary=False,
        )
        return False, bench_time, configs_timings, best_config

    def _get_config_for_key(self, key, nargs, args, kwargs):
        with self._cache_lock:
            cached = self._cache.get(key)
            if cached is not None:
                return cached, True, None

            future = self._cache_futures.get(key)
            if future is None:
                future = CacheFuture()
                self._cache_futures[key] = future
                runner = True
            else:
                runner = False

        if not runner:
            future.event.wait()
            if future.error is not None:
                raise future.error
            return future.config, future.used_cached_result, future.bench_time

        pruned_configs = self.prune_configs(kwargs, nargs)

        def benchmark():
            bench_start = time.time()
            timings = {
                config: self._bench(nargs, *args, config=config, **kwargs)
                for config in pruned_configs
            }
            bench_duration = time.time() - bench_start
            best_config = builtins_mod.min(timings, key=timings.get)
            full_nargs_local = {**nargs, **kwargs, **best_config.all_kwargs()}
            self.pre_hook(full_nargs_local, reset_only=True)
            return timings, bench_duration, best_config

        try:
            if self.cache_results:
                used_cached_result, bench_time, configs_timings, best_config = patched_check_disk_cache(
                    self, key, pruned_configs, benchmark
                )
            else:
                configs_timings, bench_time, best_config = benchmark()
                used_cached_result = False

            if configs_timings is not None:
                self.configs_timings = configs_timings
            self.bench_time = bench_time

            if best_config is not None:
                with self._cache_lock:
                    self._cache[key] = best_config

            future.config = best_config
            future.used_cached_result = used_cached_result
            future.bench_time = bench_time
            return best_config, used_cached_result, bench_time
        except BaseException as exc:
            future.error = exc
            raise
        finally:
            future.event.set()
            with self._cache_lock:
                self._cache_futures.pop(key, None)

    def patched_run(self, *args, **kwargs):
        nargs = dict(zip(self.arg_names, args))
        used_cached_result = True
        bench_time = None
        key = None
        if len(self.configs) > 1:
            all_args = {**nargs, **kwargs}
            named_args = {k: v for (k, v) in all_args.items() if k in self.arg_names}
            key_values = [named_args[name] for name in self.keys if name in named_args]
            for _, arg in named_args.items():
                if hasattr(arg, "dtype"):
                    key_values.append(str(arg.dtype))
            key = tuple(key_values)
            config, used_cached_result, bench_time = _get_config_for_key(self, key, nargs, args, kwargs)
        else:
            config = self.configs[0]

        self.cache = self._cache
        self.best_config = config
        if knobs.autotuning.print and key is not None and not used_cached_result:
            bench_time_value = bench_time if bench_time is not None else (self.bench_time or 0.0)
            print(
                f"Triton autotuning for function {self.base_fn.__name__},\n"
                f"with key as {key},\n"
                f"finished after {bench_time_value:.2f}s,\n"
                f"best config selected: {self.best_config};"
            )
        full_nargs = {**nargs, **kwargs, **config.all_kwargs()}
        if config.pre_hook is not None:
            config.pre_hook(full_nargs)
        return self.fn.run(
            *args,
            **kwargs,
            **config.all_kwargs(),
        )

    autotuner_cls.__init__ = patched_init
    autotuner_cls.check_disk_cache = patched_check_disk_cache
    autotuner_cls._get_config_for_key = _get_config_for_key
    autotuner_cls.run = patched_run
    autotuner_cls._gptqmodel_threadsafe = True
