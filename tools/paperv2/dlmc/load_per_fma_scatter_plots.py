import glob
import sys
from tools.paperv2.plot_utils import *
import matplotlib.pyplot as plt
import numpy as np
from tools.paperv2.utils import savefig
from matplotlib import rc, rcParams
from matplotlib.cm import ScalarMappable
import matplotlib.ticker as mtick   

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 15})
plt.rcParams["figure.figsize"] = (6, 4)
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0

fig, axs = plt.subplots(1, 2, figsize=(8,6))
xlim = (0,3)
ylim = (0.5,5)

df = load_dlmc_df('cascadelake/', nthreads=16)
df = filter(df, best_nano=True)
df = df[df['best_nano'] == True]
df = df[(df["sparsity"] <= 0.95) & (df["sparsity"] >= 0.6) & (df["n"] < 1024)]

df["loads_per_fma"] = df["UOPS_LOADS"] / ((df["SP_AVXW"] + df["SP_AVX"] + df["SP_SSE"] + df["SP_SINGLE"]) / 2)
df = df[df['loads_per_fma'] < 5]

ax = axs[0]
ax = df.plot.scatter(x='loads_per_fma', y='Speed-up vs Sparse', c='sparsity', colormap='cividis', alpha=0.5, s=1, ax=ax, colorbar=False)
ax.axhline(y=1.0, color='r', linestyle='-', linewidth=0.5)
#ax.axvline(x = 1.0, color = 'black', linewidth=0.5, linestyle=(0, (3, 5)), alpha=0.7)
ax.set_title('Versus MKL SpMM (CSR)', fontsize=16)
ax.set_ylabel('Speedup', fontsize=14)
ax.set_xlabel('Loads-per-FMA', fontsize=14)
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.set_ylim(ylim)
ax.set_xlim(xlim)

df = pd.concat(
    [
    pd.read_csv(RESULTS_DIR + "/cache/raspberrypi_prof/part1_per_part.csv"),
    pd.read_csv(RESULTS_DIR + "/cache/raspberrypi_prof/part2_per_part.csv"),
    ]
)

df = filter(df, best_nano=True, numThreads=4)
df = df[df['best_nano'] == True]
df = df[(df["sparsity"] <= 0.95) & (df["sparsity"] >= 0.6) & (df["n"] < 1024)]

df["SP_FLOPS_TOTAL"] = df['gflops'] / 8
df["loads_per_fma"] = df["LOADS"] / ((df["SP_FLOPS_TOTAL"]) / 2)
df = df[df['loads_per_fma'] < 5]

ax = axs[1]
ax = df.plot.scatter(x='loads_per_fma', y='Speed-up vs Sparse', c='sparsity', colormap='cividis', alpha=0.5, s=1, ax=ax, colorbar=False)
ax.axhline(y=1.0, color='r', linestyle='-', linewidth=0.5)
#ax.axvline(x = 1.0, color = 'black', linewidth=0.5, linestyle=(0, (3, 5)), alpha=0.7)
ax.set_title('Versus XNNPACK', fontsize=16)
ax.set_ylabel(None)
ax.set_xlabel('Load-per-FMA', fontsize=14)
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.set_ylim(ylim)
ax.set_xlim(xlim)

cmap = plt.get_cmap("cividis")
norm = plt.Normalize(60, 95)
sm =  ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axs, pad=0.15, location='bottom', shrink=0.45)
cbar.ax.set_title("Sparsity", position=(-0.15,5), pad=-4, y=0.3, fontsize=14)
cbar.ax.tick_params(axis='both', which='major', labelsize=14)
cbar.ax.tick_params(axis='both', which='minor', labelsize=8)
cbar.ax.xaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
cbar.ax.set_xticks([x + 5 for x in cbar.ax.get_xticks()[:-1]])

savefig("loads_per_fma_scatter.pdf")