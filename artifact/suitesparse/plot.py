import glob
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
from tools.paper.plotting.plot_utils import plot_save
from matplotlib import rc, rcParams
from tools.paper.plotting.plot_utils import *
from scipy.stats import gmean

from artifact.utils import *
from artifact.suitesparse.post_process_results import get_df, thread_list, bcols_list, ARCH

from matplotlib.ticker import StrMethodFormatter, NullFormatter

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 15})
plt.rcParams["figure.figsize"] = (6, 4)
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0

def speedup_column(method, baseline):
    return f'Speed-up {method} vs. {baseline}'

def compute_speedup(df, method, baseline):
    df[speedup_column(method, baseline)] = df[f"time median|{baseline}"] / df[f"time median|{method}"]

MARKSIZE = 1

colors = {
    32: 'DarkBlue',
    128: 'DarkRed'
}
ax = None
for bcols in [32, 128]:
    df = get_df(bcols, 20)
    compute_speedup(df, "Sp. Reg.", "MKL_Sparse")
    ax = df.plot(kind='scatter', x="NNZ", y=speedup_column("Sp. Reg.", "MKL_Sparse"), c=colors[bcols], ax=ax, s=MARKSIZE)
plt.tight_layout()
plt.savefig(PLOTS_DIR + f"/suitesparse_{ARCH}.jpg")
print("Created:", PLOTS_DIR + f"/suitesparse_{ARCH}.jpg")

ax = None
for bcols in [128]:
    df = get_df(bcols, 20)
    df["sparsity_raw"] = df["sparsity_raw"]
    df["density"] = 1 - df["sparsity_raw"] 
    compute_speedup(df, "Sp. Reg.", "MKL_Sparse")
    ax = df.plot(kind='scatter', x="density", y=speedup_column("Sp. Reg.", "MKL_Sparse"), c='cov|Sp. Reg.', ax=ax, s=MARKSIZE, norm=matplotlib.colors.LogNorm())
ax.set(xscale="log", yscale="linear")
plt.tight_layout()
plt.xlabel('Density (Log)')
plt.ylim(0, 7)
plt.axhline(y = 1.0, color = 'r', linestyle = '-')
plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:g}'))
plt.gca().xaxis.set_minor_formatter(NullFormatter())

plt.savefig(PLOTS_DIR + f"/suitesparse_{ARCH}_density_log.pdf")
print("Created:", PLOTS_DIR + f"/suitesparse_{ARCH}_density_log.pdf")

ax = None
for bcols in [32, 128]:
    df = get_df(bcols, 20)
    df["sparsity_raw"] = df["sparsity_raw"]
    compute_speedup(df, "Sp. Reg.", "MKL_Sparse")
    ax = df.plot(kind='scatter', x="sparsity_raw", y=speedup_column("Sp. Reg.", "MKL_Sparse"), c=colors[bcols], ax=ax, s=MARKSIZE)
plt.tight_layout()
plt.xlabel('Sparsity')
plt.axhline(y = 1.0, color = 'r', linestyle = '-')
plt.savefig(PLOTS_DIR + f"/suitesparse_{ARCH}_sparsity.jpg")
print("Created:", PLOTS_DIR + f"/suitesparse_{ARCH}_sparsity.jpg")


ax = None
for bcols in [128]:
    df = get_df(bcols, 20)
    df["sparsity_raw"] = df["sparsity_raw"]
    for matrixPath in df[df["sparsity_raw"] <= 0.9]["Matrix"]:
        print(matrixPath)

    df["density"] = 1 - df["sparsity_raw"] 
    compute_speedup(df, "Sp. Reg.", "MKL_Sparse")
    ax = df.plot(kind='scatter', x="cov|Sp. Reg.", c='density', y=speedup_column("Sp. Reg.", "MKL_Sparse"), ax=ax, s=MARKSIZE, norm=matplotlib.colors.LogNorm())
plt.tight_layout()
plt.xlabel('COV')
plt.axhline(y = 1.0, color = 'r', linestyle = '-')
plt.ylim(0, 4)
ax.set(xscale="log", yscale="linear")
plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:g}'))
plt.gca().xaxis.set_minor_formatter(NullFormatter())
plt.savefig(PLOTS_DIR + f"/suitesparse_{ARCH}_cov.jpg")
print("Created:", PLOTS_DIR + f"/suitesparse_{ARCH}_cov.jpg")