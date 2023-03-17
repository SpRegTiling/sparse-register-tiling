RESULTS_DIR = "/sdb/paperv2_results/"
CACHEV2_DIR = "/sdb/paperv2_results/cachev2/"
PLOTS_DIR = "/workspaces/spmm-nano-bench/plots/v2/"

import os
os.makedirs(PLOTS_DIR, exist_ok=True)

def filter(df, **kwargs):
    bool_index = None
    for key, value in kwargs.items():
        if isinstance(value, list):
            _bool_index = df[key].isin(value)
        else:
            _bool_index = df[key] == value
        if bool_index is None:
            bool_index = _bool_index
        else:
            bool_index = bool_index & _bool_index
    return df[bool_index]
