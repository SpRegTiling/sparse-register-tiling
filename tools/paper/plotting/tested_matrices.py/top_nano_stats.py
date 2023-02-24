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

sparsity_buckets = pd.IntervalIndex.from_tuples([(0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)])
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

for top_nano in top_nanos:
    dff = filter(df, name=top_nano)
    print()
    print(top_nano)
    print(dff["sparsity_buckets"].value_counts())
    print(dff["n"].value_counts())
    print(list(dff.columns))

print(df["mapping"].unique())
print(df["Mr"].unique())
print(df["Nr"].unique())
