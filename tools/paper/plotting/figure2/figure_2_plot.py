import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
import sys; sys.path.insert(0,f'{SCRIPT_DIR}/../')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.api.types import is_string_dtype

import altair as alt
import altair_saver
import os
import post_process

from plot_utils import *
from cache_utils import cached_merge_and_load, cache_df_processes
from collections import defaultdict
from tools.plotting.color import divergent_color_scheme
from scipy.stats.mstats import gmean

PLOT_DIR = SCRIPT_DIR + "/plots/"
os.makedirs(PLOT_DIR, exist_ok=True)


alt.data_transformers.enable('default', max_rows=1000000)
unique_config_columns = ["name", "n", "m_tile", "k_tile", "n_tile"]

PLOT_GFLOPS = False
PLOT_SPEEDUP = True
color_scheme = 'purpleblue'

cluster = 'niagara'
df = pd.read_csv(RESULTS_DIR + '/figure2/figure_2_merged_postprocessed.csv')

b_colss = sorted(df["n"].unique())
n_threadss = df["numThreads"].unique()
methods = df["name"].unique()
print(methods)
print(df.columns)

methods =['MKL_Dense mkl', 'MKL_Sparse', 'MKL_Sparse_IE', 'NANO_Best', 'ASpT']

pruning_methods = [
    'random_pruning', 'variational_dropout'
]

matrix_path = '/gpfs/fs0/scratch/m/mmehride/lcwilkin/2/random/random_80_768_768.mtx'


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


df["loads_per_fma"] = df["L1_DCA"] / ((df["SP_AVXW"] + df["SP_AVX"] + df["SP_SSE"] + df["SP_SINGLE"]) / 2)


def compute_nano(x):
    x["best_nano"] = False
    nanos = x[x["name"].str.contains("NANO")]
    if not nanos.empty:
        x.loc[nanos["time median"].idxmin(), "best_nano"] = True
    return x


def compute_nano_sub_1(x):
    x["best_sub1_nano"] = False
    nanos = x[(x["name"].str.contains("NANO")) & (x["loads_per_fma"] <= 1)]
    if not nanos.empty:
        x.loc[nanos["time median"].idxmin(), "best_nano"] = True
    return x

print("computing for groups ...")
df = post_process.compute_for_group(df,
                                    [compute_nano, compute_nano_sub_1],
                                    group_by=["matrixPath", "n", "numThreads"])


print(df["n"].unique())
df["is_nano"] = df["name"].str.contains("NANO")
df["include"] = df["name"].isin(
    ["ASpT", "MKL_BSR_B8", "MKL_Dense", "MKL_Sparse", "MKL_Dense "]
) | (df["is_nano"] & (df["best_nano"] | df["best_sub1_nano"]))

df = filter(df, matrixPath=matrix_path, numThreads=1, n=256, include=True)
# Divide by 2, papi counts FMA as 2 flops
df["flops"] = df["n"] * df["nnz"]
print(df["flops"])
print(df["SP_FLOPS_TOTAL"].min())

df.loc[df['is_nano'], "name"] = "Sparse Register Tiling"
df.loc[df['name'] == 'MKL_BSR_B8', "name"] = "Register Blocking, (MKL BSR)"
df.loc[df['name'] == 'MKL_Sparse', "name"] = "MKL Sparse (CSR)"
df.loc[df['name'] == 'MKL_Dense ', "name"] = "MKL Dense"

"""
Model name:            Intel(R) Xeon(R) Gold 6248 CPU @ 2.50GHz
Stepping:              7
CPU MHz:               3200.073
CPU max MHz:           3900.0000
CPU min MHz:           1000.0000
"""

freq = 3_200_000_000
compute_bound_min_time = (df["SP_FLOPS_TOTAL"].min() / (freq*2*16)) * 1_000_000
load_bound_min_time = ((df["SP_FLOPS_TOTAL"].min() / (freq*2*16)) *2.5) * 1_000_000

chart = alt.Chart(df).mark_point().encode(
    x=alt.X('loads_per_fma', title='load \u00B5ops / fma \u00B5ops'),
    y=alt.Y('time median', title='Execution Time (\u00B5s)'),
    size=alt.Size('SP_FLOPS_TOTAL', title='Total FLOPs'),
    color='name'
)

text = alt.Chart(df).mark_text(
    align='left',
    baseline='middle',
    dx=10
).encode(
    x=alt.X('loads_per_fma'),
    y=alt.Y('time median'),
    text='name'
)

line1 = alt.Chart(
    pd.DataFrame({'loads_per_fma': [0, 1], 'time median': [compute_bound_min_time, compute_bound_min_time]})).mark_line().encode(
    alt.X('loads_per_fma'),
    alt.Y('time median'),
)
line2 = alt.Chart(
    pd.DataFrame({'loads_per_fma': [1, 2.5], 'time median': [compute_bound_min_time, load_bound_min_time]})).mark_line().encode(
    alt.X('loads_per_fma'),
    alt.Y('time median'),
)

(chart + line1 + line2 + text).show()
print(len(df))