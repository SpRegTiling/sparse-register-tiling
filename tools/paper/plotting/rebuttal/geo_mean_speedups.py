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


def tabluarize(df_tab):
    # vs_dense = ["Speed-up vs Dense"]
    # vs_sparse = ["Speed-up vs Sparse"]
    # vs_dense_percent = ["Percent Faster vs Dense"]
    # vs_sparse_percent = ["Percent Faster vs Sparse"]
    headers = ["Sparsity Range"]
    vs_sparse_percent = ["Percent Faster vs Sparse"]

    cat_pairs = [(str(x), x) for x in df["sparsity_buckets"].unique()]
    cat_pairs = [(x[0], x[1]) for x in cat_pairs if x[0] != 'nan']
    cat_pairs = sorted(cat_pairs, key=lambda x: x[0])
    cat_strs = [x[0] for x in cat_pairs]
    cats = []
    for i in range(len(cat_strs)):
        cats.append([])

    for i, s in enumerate(cat_strs):
        cats[i].append(s)

    # vs_dense.append(stats.gmean(df_tab["Speed-up vs Dense"]))
    # vs_sparse.append(stats.gmean(df_tab["Speed-up vs Sparse"]))
    # vs_dense_percent.append(len(df_tab[df_tab["Speed-up vs Dense"] > 1]) / len(df_tab))
    # vs_sparse_percent.append(len(df_tab[df_tab["Speed-up vs Sparse"] > 1]) / len(df_tab))

    for bcols in sorted(df_tab["n"].unique()):
        df_bcols = df_tab[df_tab["n"] == bcols]
        headers.append(bcols)
        for i, (s, k) in enumerate(cat_pairs):
            dff = df_bcols[df_bcols["sparsity_buckets"] == k]
            cats[i].append(dff["gflops/s"].mean())

        # vs_dense.append(stats.gmean(df_bcols["Speed-up vs Dense"]))
        # vs_sparse.append(stats.gmean(df_bcols["Speed-up vs Sparse"]))
        # vs_dense_percent.append(len(df_bcols[df_bcols["Speed-up vs Dense"] > 1]) / len(df_bcols))
        # vs_sparse_percent.append(len(df_bcols[df_bcols["Speed-up vs Sparse"] > 1]) / len(df_bcols))

    return tabulate(cats, headers=headers)


df = load_dlmc_df(SUBFOLDER, nthreads=20)
df = df[(df["sparsity"] <= 0.95) & (df["sparsity"] >= 0.6) & (df["n"] < 1024)]
df = df.reset_index(drop=True)


sparsity_buckets = pd.IntervalIndex.from_tuples([(0.0, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)])
df["sparsity_buckets"] = pd.cut(df["sparsity"], sparsity_buckets)
df["flops"] = 2 * df["n"] * df["nnz"]
df["gflops/s"] = (df["flops"] / (df["time median"]/1e6)) / 1e9
df["rflops"] = 2 * df["n"] * df["m"] * df["k"]
df["rgflops/s"] = (df["rflops"] / (df["time median"]/1e6)) / 1e9

print()

print("Single-threaded speedup")
print("Num matrices", df["matrixId"].nunique())
print("Bcols", df["n"].unique())

for method in list(df["name"].unique()):
    if "NANO" in method:
        print()
        print(method)
        df_filt = df[(df["name"] == method)]
        print(tabluarize(df_filt))

for method in list(df["name"].unique()):
    if "MKL" in method:
        print()
        print(method)
        df_filt = df[(df["name"] == method)]
        print(tabluarize(df_filt))

for method in list(df["name"].unique()):
    if "ASpT" in method:
        print()
        print(method)
        df_filt = df[(df["name"] == method)]
        print(tabluarize(df_filt))


df_filt = filter(df, best_aspt=True, num_aspt=2)
print()
print(df_filt["matrixPath"].nunique())
print("Best ASpT")
print(tabluarize(df_filt))

df_filt = filter(df, best_nano=True)
print()
print(df_filt["matrixPath"].nunique())
print("Best Nano")
print(tabluarize(df_filt))

