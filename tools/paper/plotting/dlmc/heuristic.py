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


def report_top(df_tab):
    headers = ["Sparsity Range"]

    cat_pairs = [(str(x), x) for x in df["sparsity_buckets"].unique()]
    cat_pairs = [(x[0], x[1]) for x in cat_pairs if x[0] != 'nan']
    cat_pairs = sorted(cat_pairs, key=lambda x: x[0])
    cat_strs = [x[0] for x in cat_pairs]
    cats = []

    for cat, cat_str in cat_pairs:
        for bcols in sorted(df_tab["n"].unique()):
            dff = df_tab[(df_tab["sparsity_buckets"] == cat) & (df_tab[df_tab["n"] == bcols])]
            print(cat_str, bcols, dff["name"].value_counts().index.tolist()[:2])


def tabluarize(df_tab):
    headers = ["Sparsity Range"]

    cat_pairs = [(str(x), x) for x in df["sparsity_buckets"].unique()]
    cat_pairs = [(x[0], x[1]) for x in cat_pairs if x[0] != 'nan']
    cat_pairs = sorted(cat_pairs, key=lambda x: x[0])
    cat_strs = [x[0] for x in cat_pairs]
    cats = []
    for i in range(len(cat_strs)):
        cats.append([])

    for i, s in enumerate(cat_strs):
        cats[i].append(s)

    for bcols in sorted(df_tab["n"].unique()):
        df_bcols = df_tab[df_tab["n"] == bcols]
        headers.append(bcols)
        for i, (s, k) in enumerate(cat_pairs):
            dff = df_bcols[df_bcols["sparsity_buckets"] == k]
            cats[i].append(dff["gflops/s"].mean())

    return tabulate(cats, headers=headers)

df = load_dlmc_df(SUBFOLDER, nthreads=16)
df = df[(df["sparsity"] <= 0.95) & (df["sparsity"] >= 0.6) & (df["n"] < 1024)]
df = df.reset_index(drop=True)

print("Single-threaded speedup")
print("Num matrices", df["matrixId"].nunique())
print("Bcols", df["n"].unique())

sparsity_buckets = pd.IntervalIndex.from_tuples([(0.0, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)])
df["sparsity_buckets"] = pd.cut(df["sparsity"], sparsity_buckets)
df["flops"] = 2 * df["n"] * df["nnz"]
df["gflops/s"] = (df["flops"] / (df["time median"]/1e6)) / 1e9

# for method in list(df["name"].unique()):
#     if "NANO" not in method: continue
#     print()
#     print(method)
#     df_filt = df[(df["name"] == method)]
#     print(tabluarize(df_filt))


# for method in list(df["name"].unique()):
#     if "NANO" in method: continue
#     print()
#     print(method)
#     df_filt = df[(df["name"] == method)]
#     print(tabluarize(df_filt))

print(report_top(df))

