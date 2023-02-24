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

dfs = []

def ASpT_correct(x):
    runs = x[x["name"] == 'ASpT']
    x[f'aspt_correct'] = False
    if not runs.empty and runs.iloc[0]["correct"] == "correct":
        x[f'aspt_correct'] = True
    return x

def HasBoth(x):
    run_aspt = x[x["name"] == 'ASpT']
    run_mkl = x[x["name"] == 'MKL_Sparse']
    x[f'has_both'] = False
    if not run_mkl.empty and not run_aspt.empty and run_aspt.iloc[0]["aspt_correct"]:
        x[f'has_both'] = True
    return x

def HasBothBcols(x):
    x[f'has_both_bcols'] = False
    runs_32 = x[x["n"] == 32]
    runs_128 = x[x["n"] == 128]
    if not runs_32.empty and not runs_128.empty:
        if runs_32.iloc[0]["has_both"] and runs_128.iloc[0]["has_both"]:
            x[f'has_both_bcols'] = True
    return x

for method in ["mkl", "aspt"]:
    for bcols_str in ["small"]:
        df = pd.read_csv(RESULTS_DIR + f"rebuttal/aspt_aspt/{ss}_AVX512_{ss}_AVX512_{method}_{bcols_str}_bcols_20_32.csv")
        dfs.append(df)
    df["gflops"] = 2 * df["n"] * df["nnz"]
    df["gflops/s"] = (df["gflops"] / (df["time median"] / 1e6)) / 1e9

    df["matrixName"] = df["matrixPath"].str.split("data").str[-1]
df = pd.concat(dfs)
df = df.groupby(["matrixPath", "n", "numThreads"], group_keys=False).apply(ASpT_correct).reset_index(drop=True)
df = df.groupby(["matrixPath", "n", "numThreads"], group_keys=False).apply(HasBoth).reset_index(drop=True)
df = df.groupby(["matrixPath", "numThreads"], group_keys=False).apply(HasBothBcols).reset_index(drop=True)


print("=======", ss, "========")
for bcols in [32, 128]:
    print("BCols", bcols, "========")
    for name in df["name"].unique():
        print(ss, method, name)
        df_20 = filter(df, name=name, numThreads=20, n=bcols, has_both_bcols=True)
        print("num mtx:", len(df_20))
        print("   mean:", df_20["gflops/s"].mean())
        print("   min :", df_20["gflops/s"].min())
        print("   max :", df_20["gflops/s"].max())


for bcols in [32, 128]:
    dff = filter(df, name="ASpT",numThreads=20, n=bcols, has_both_bcols=True)
    dfw = pd.pivot(dff, index=["matrixName", "m", "k", "nnz", "n"], columns=["name"],
                   values=["gflops/s"])

    dfw.columns = dfw.columns.get_level_values(level=1)
    dfw.index.names = ['Matrix', "Rows", "Cols", "NNZ", "Bcols"]
    dfw.to_csv(f'gflops_aspt_suitesparse_20_thread_bcols_{bcols}.csv')
