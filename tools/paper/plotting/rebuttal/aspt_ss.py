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

SUBFOLDER = sys.argv[1] + '/'
NTHREADS = {
    'cascadelake/': 20,
    'raspberrypi/': 4,
}[SUBFOLDER]

BASLINE = {
    'cascadelake/': 20,
    'raspberrypi/': 4,
}[SUBFOLDER]


def tabluarize(df_tab):
    vs_dense = ["Speed-up vs Dense"]
    vs_sparse = ["Speed-up vs Sparse"]
    vs_dense_percent = ["Percent Faster vs Dense"]
    vs_sparse_percent = ["Percent Faster vs Sparse"]
    headers = ["Name", "gmean"]

    vs_dense.append(stats.gmean(df_tab["Speed-up vs Dense"]))
    vs_sparse.append(stats.gmean(df_tab["Speed-up vs Sparse"]))
    vs_dense_percent.append(len(df_tab[df_tab["Speed-up vs Dense"] > 1]) / len(df_tab))
    vs_sparse_percent.append(len(df_tab[df_tab["Speed-up vs Sparse"] > 1]) / len(df_tab))

    for bcols in sorted(df_tab["n"].unique()):
        df_bcols = df_tab[df_tab["n"] == bcols]
        headers.append(bcols)
        vs_dense.append(stats.gmean(df_bcols["Speed-up vs Dense"]))
        vs_sparse.append(stats.gmean(df_bcols["Speed-up vs Sparse"]))
        vs_dense_percent.append(len(df_bcols[df_bcols["Speed-up vs Dense"] > 1]) / len(df_bcols))
        vs_sparse_percent.append(len(df_bcols[df_bcols["Speed-up vs Sparse"] > 1]) / len(df_bcols))

    return tabulate([vs_dense, vs_sparse, vs_dense_percent, vs_sparse_percent], headers=headers)


for ss in ["ss_small", "ss_large"]:
    print("=======", ss, "========")
    for bcols in [32, 128]:
        print("BCols", bcols, "========")
        for method in ["mkl", "aspt"]:
            dfs = []
            for bcols_str in ["small"]:
                df = pd.read_csv(RESULTS_DIR + f"rebuttal/aspt_ss/{ss}_AVX512_{ss}_AVX512_{method}_{bcols_str}_bcols_20_32.csv")
                dfs.append(df)
            df = pd.concat(dfs)
            df["gflops"] = 2 * df["n"] * df["nnz"]
            df["gflops/s"] = (df["gflops"] / (df["time median"] / 1e6)) / 1e9

            for name in df["name"].unique():
                print(ss, method, name)
                df_20 = filter(df, name=name, numThreads=20, correct="correct", n=bcols)
                print("  mean:", df_20["gflops/s"].mean())
                print("  min :", df_20["gflops/s"].min())
                print("  max :", df_20["gflops/s"].max())
