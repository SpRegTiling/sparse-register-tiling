import glob
import sys
import matplotlib.pyplot as plt
import numpy as np
from tools.paper.plotting.plot_utils import plot_save
from matplotlib import rc, rcParams
from tools.paper.plotting.plot_utils import *
from scipy import stats
from tabulate import tabulate

SUBFOLDER = sys.argv[1] + '/'
NTHREADS = {
    'cascadelake/': 16,
    'raspberrypi/': 4,
}[SUBFOLDER]

BASLINE = {
    'cascadelake/': 16,
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


df = load_dlmc_df(SUBFOLDER, nthreads=16)
df = df[(df["sparsity"] <= 0.95) & (df["sparsity"] >= 0.6) & (df["n"] < 1024)]
df = df.reset_index(drop=True)

df["flops"] = 2 * df["n"] * df["nnz"]
df["rflops"] = 2 * df["n"] * df["m"] * df["k"]
df["rgflops/s"] = (df["rflops"] / (df["time median"]/1e6)) / 1e9

print(df["rgflops/s"].value_counts())
print(df["name"].unique())
print(filter(df, n=128, name="MKL_Dense mkl", numThreads=16)["rgflops/s"].mean(skipna=True))
print(filter(df, n=128, name="MKL_Dense mkl", numThreads=16)["rgflops/s"].min(skipna=True))
print(filter(df, n=128, name="MKL_Dense mkl", numThreads=16)["rgflops/s"].max(skipna=True))


# print("Single-threaded speedup")
# print("Num matrices", df["matrixId"].nunique())
# print("Bcols", df["n"].unique())
#
# for method in list(df["name"].unique()):
#     if "NANO" not in method: continue
#     print()
#     print(method)
#     df_filt = df[(df["name"] == method)]
#     print(tabluarize(df_filt))
#
#
# for method in list(df["name"].unique()):
#     if "NANO" in method: continue
#     print()
#     print(method)
#     df_filt = df[(df["name"] == method)]
#     print(tabluarize(df_filt))
#
# print()
# print("Best Nano Counts")
# df = filter(df, best_nano=True)
# print(df["name"].value_counts())
#
# print()
# print("Best Nano")
# print(tabluarize(df))

