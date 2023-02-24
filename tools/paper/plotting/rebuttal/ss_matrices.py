import glob
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tools.paper.plotting.plot_utils import plot_save
from matplotlib import rc, rcParams
from tools.paper.plotting.plot_utils import *
from scipy import stats
from tabulate import tabulate

ss = "aspt_mtxs"


def compute_best_nano(x):
    x["best_nano"] = False
    nanos = x[x["is_nano"] == True]
    x["num_nano"] = len(nanos)
    if not nanos.empty:
        x.loc[x['time median'] == nanos['time median'].min(), "best_nano"] = True
    return x

def has(x):
    x["has_best_nano"] = x["best_nano"].any()
    return x

def HasBothBcols(x):
    x[f'has_both_bcols'] = False
    runs_32 = x[x["n"] == 32]
    runs_128 = x[x["n"] == 128]
    if not runs_32.empty and not runs_128.empty:
        if runs_32.iloc[0]["has_both"] and runs_128.iloc[0]["has_both"]:
            x[f'has_both_bcols'] = True
    return x

files = [
    "ss_small_AVX512_ss_small_AVX512_nano8_bests_part1_small_bcols_20_32.csv",
    "ss_small_AVX512_ss_small_AVX512_nano4_bests_part1_small_bcols_20_32.csv",
    "ss_small_AVX512_ss_small_AVX512_nano8_bests_part2_small_bcols_20_32.csv",
    "ss_small_AVX512_ss_small_AVX512_nano4_bests_part2_small_bcols_20_32.csv",
    "ss_small_AVX512_ss_small_AVX512_mkl_small_bcols_20_32.csv",
]

dfs = []
for file in files:
    dfs.append(pd.read_csv(RESULTS_DIR + f"rebuttal/ss/{file}", index_col=False))
df = pd.concat(dfs)

print(df)
print(df['name'])
nano_methods = df['name'].str.contains(r'M[0-9]N[0-9]', regex=True)
df.loc[nano_methods, 'name'] = "NANO_" + df.loc[nano_methods, 'name']
df["is_nano"] = df['name'].str.contains("NANO")

df["matrixName"] = df["matrixPath"].str.split("ss").str[-1]
df = df.groupby(["matrixPath", "n", "numThreads"], group_keys=False).apply(compute_best_nano).reset_index(drop=True)
df = df.groupby(["matrixPath", "n", "numThreads"], group_keys=False).apply(has).reset_index(drop=True)

df["gflops"] = 2 * df["n"] * df["nnz"]
df["gflops/s"] = (df["gflops"] / (df["time median"] / 1e6)) / 1e9

df = df[(df['best_nano'] == True) | df['name'].str.contains('MKL')]
df["Method"] = df["name"]
df.loc[df['best_nano'] == True, "Method"] = "Sp. Reg."


for bcols in [32, 128]:
    dff = filter(df, numThreads=20, n=bcols, has_best_nano=True, correct="correct")
    dfw = pd.pivot(dff, index=["matrixName", "m", "k", "nnz", "n"], columns=["Method"],
                   values=["gflops/s"])

    dfw.columns = dfw.columns.get_level_values(level=1)
    dfw.index.names = ['Matrix', "Rows", "Cols", "NNZ", "Bcols"]
    dfw.to_csv(f'suitesparse_bcols_{bcols}.csv')