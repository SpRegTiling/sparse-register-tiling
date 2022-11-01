import torch
import os
import pandas as pd

DEFAULT_CACHE_DIR='/sdb/cache'


def cache_dataframe(cache_dir=DEFAULT_CACHE_DIR, refresh_cache=False):
    def decorator(fn):
        def wrap(*args, **kwargs):
            cache_file = cache_dir + "/" + fn.__name__ + ".cache.csv"
            if os.path.exists(cache_file) and not refresh_cache:
                return pd.read_csv(cache_file)

            df = fn(*args, **kwargs)
            df.to_csv(cache_file)
            return df
        return wrap
    return decorator


def cached_return_by_input_id(cache_dir=DEFAULT_CACHE_DIR, refresh_cache=False, suffix=None, kwargs_to_check=None):
    def decorator(fn):
        def wrap(input, **kwargs):
            _suffix = suffix if suffix is not None else fn.__name__
            if kwargs_to_check is not None:
                _suffix += "_" + "_".join([str(kwargs.get(x, "")) for x in kwargs_to_check])

            cache_file = f'{cache_dir}/{input._cache_id}_{_suffix}.cache.pt'
            if os.path.exists(cache_file) and not refresh_cache:
                return torch.load(cache_file)

            ret = fn(input, **kwargs)
            torch.save(ret, cache_file)
            return ret
        return wrap
    return decorator


def cache_dataframe(cache_dir=DEFAULT_CACHE_DIR, refresh_cache=False):
    def decorator(fn):
        def wrap(*args, **kwargs):
            cache_file = cache_dir + "/" + fn.__name__ + ".cache.csv"
            if os.path.exists(cache_file) and not refresh_cache:
                return pd.read_csv(cache_file)

            df = fn(*args, **kwargs)
            df.to_csv(cache_file)
            return df
        return wrap
    return decorator


def cached_return(cache_dir=DEFAULT_CACHE_DIR, refresh_cache=False):
    def decorator(fn):
        def wrap(*args, **kwargs):
            cache_file = cache_dir + "/" + fn.__name__ + ".cache.csv"
            if os.path.exists(cache_file) and not refresh_cache:
                return torch.load(cache_file)

            ret = fn(*args, **kwargs)
            torch.save(ret, cache_file)
            return ret
        return wrap
    return decorator
