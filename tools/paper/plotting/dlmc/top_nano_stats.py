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

df = load_dlmc_df(SUBFOLDER, nthreads=16)
df = df[(df["sparsity"] <= 0.95) & (df["sparsity"] >= 0.6) & (df["n"] < 1024)]
df = df.reset_index(drop=True)

df = filter(df, best_nano=True)

sparsity_buckets = pd.IntervalIndex.from_tuples([(0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 0.95)])
df["sparsity_buckets"] = pd.cut(df["sparsity"], sparsity_buckets)


top_nanos = [
    "NANO_M4N4_NKM_LB_SA_identity",
    "NANO_M4N4_KNM_LB_TLB128_SA_identity",
    "NANO_M4N4_NKM_LB_TLB128_SA_identity",
    "NANO_M8N2_KNM_LB_TLB64_SA_alt",
    "NANO_M8N2_KNM_orig",
    "NANO_M8N2_NKM_alt",
    "NANO_M4N4_KNM_identity",
    "NANO_M8N2_NKM_LB_TLB64_SA_alt",
    "NANO_M8N3_KNM_LB_TLB128_SA_orig",
    "NANO_M8N2_NKM_orig",
    "NANO_M8N3_KNM_LB_orig",
    "NANO_M8N2_KNM_alt",
    "NANO_M4N4_NKM_LB_TLB64_SA_identity",
    "NANO_M4N4_KNM_LB_SA_identity",
    "NANO_M4N4_NKM_LB_orig",
    "NANO_M8N3_NKM_LB_TLB128_SA_orig",
    "NANO_M8N2_KNM_LB_TLB128_SA_orig",
    "NANO_M4N4_NKM_LB_TLB64_SA_orig",
    "NANO_M4N4_NKM_identity",
    "NANO_M4N4_KNM_LB_orig",
]

df["sparsity_real"] = round(1- df["nnz"] / (df["k"] * df["m"]), 2)

def dump(m):
    print(m["matrixPath"], m["n"], m["sparsity"], m["sparsity_buckets"], m["sparsity_real"], m['mapping'], m['Mr'])


dff = filter(df, name="NANO_M8N3_KNM_LB_orig", best_nano=True)
m = dff.iloc[0]
dump(m)
m = dff.iloc[2]
dump(m)

dff = filter(df, name="NANO_M4N4_NKM_LB_SA_identity", best_nano=True)
m = dff.iloc[2]
dump(m)
m = dff.iloc[6]
dump(m)

dff = filter(df, name="NANO_M4N4_NKM_LB_orig", best_nano=True)
m = dff.iloc[2]
dump(m)
m = dff.iloc[6]
dump(m)


dff = filter(df, name="NANO_M8N2_KNM_LB_TLB64_SA_alt", best_nano=True)
m = dff.iloc[2]
dump(m)
m = dff.iloc[6]
dump(m)

dff = filter(df, sparsity=0.6,  best_nano=True)
m = dff.iloc[6]
dump(m)