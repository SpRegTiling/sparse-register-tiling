import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
from collections import defaultdict
import hashlib
import pandas as pd
import glob

CACHE_DIR = SCRIPT_DIR + "/.cache/"
os.makedirs(CACHE_DIR, exist_ok=True)


def cached_merge_and_load(files, name, afterload_hook=None, force_use_cache=False):
    bcols_charts = defaultdict(lambda: [])
    numThreads_charts = defaultdict(lambda: [])

    existing_cache_files = glob.glob(CACHE_DIR + f'/{name}_*.csv')
    if force_use_cache and len(existing_cache_files) > 0:
        cache_file = existing_cache_files[0]
    else:
        time_sum = ""
        for file in files:
            time_sum += str(os.path.getmtime(file))

        _hash = hashlib.md5(time_sum.encode("utf8")).hexdigest()[-5:]
        cache_file = CACHE_DIR + f"/{name}_{_hash}.csv"

    print(cache_file)
    if os.path.exists(cache_file):
        df = pd.read_csv(cache_file)
        return df, False
    else:
        dfs = []
        for file in files:
            df = pd.read_csv(file)
            if afterload_hook:
                df = afterload_hook(file, df)

            dfs.append(df)
        df = pd.concat(dfs)

        for old_cache in existing_cache_files:
            try:
                os.remove(old_cache)
            except:
                print("Error while deleting old_cache: ", old_cache)

        df.to_csv(cache_file)
        return df, True


def cache_df_processes(name):
    def _cache_df_processes(fn):
        cache_file = CACHE_DIR + f"{name}.csv"

        def wrap(df, reload):
            if not reload and os.path.exists(cache_file):
                return pd.read_csv(cache_file)
            df = fn(df)
            df.to_csv(cache_file)
            return df
        return wrap
    return _cache_df_processes