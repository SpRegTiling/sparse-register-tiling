import glob
import sys
import matplotlib.pyplot as plt
import numpy as np
from tools.paper.plotting.plot_utils import plot_save
from matplotlib import rc, rcParams
from tools.paper.plotting.plot_utils import *
from scipy import stats

SUBFOLDER = sys.argv[1] + '/'
NTHREADS = {
    'cascadelake/': 16,
    'raspberrypi/': 4,
}[SUBFOLDER]

BASLINE = {
    'cascadelake/': 16,
    'raspberrypi/': 4,
}[SUBFOLDER]

df = load_dlmc_df(SUBFOLDER, nthreads=NTHREADS)

print(df['gflops'].min())

df = filter(df, best_nano=True)
df = df[(df["sparsity"] <= 0.95) & (df["sparsity"] >= 0.6) & (df["n"] < 1024)]
df = df[df["Speed-up vs Dense"] > 0.0]

# ax = df.plot.scatter(x='gflops', y='Speed-up vs Sparse', c='sparsity', colormap='cividis', alpha=0.5, s=1)
# ax.set_xscale('log')
# ax.axhline(y=1.0, color='r', linestyle='-')
#
# plt.ylabel('Speed-up vs MKL Sparse')
# plt.xlabel('Problem Size (GFLOPs)')
# plot_save(f"scatters/{SUBFOLDER}/vs_sparse")

plt.scatter(x=rand_jitter(df["sparsity"]), y=df["Speed-up vs Sparse"], alpha=0.5, s=1)
plt.gca().axhline(y=1.0, color='r', linestyle='-')

plt.ylabel('Speed-up vs MKL Sparse')
plt.xlabel('Problem Size (GFLOPs)')
plot_save(f"scatters/{SUBFOLDER}/vs_sparse_jitter")


ax = df.plot.scatter(x='gflops', y='Speed-up vs Dense', c='sparsity', colormap='cividis', alpha=0.5, s=1)
ax.set_xscale('log')
ax.axhline(y=1.0, color='r', linestyle='-')

plt.ylabel('Speed-up vs MKL Dense')
plt.xlabel('Problem Size (GFLOPs)')
plot_save(f"scatters/{SUBFOLDER}/vs_dense")

print("Num matrices", df["matrixId"].nunique())
print("Bcols", df["n"].unique())

print("Speed-up vs Dense", stats.gmean(df["Speed-up vs Dense"]))
print("Percent Faster", len(df[df["Speed-up vs Dense"] > 1]) / len(df))
print("Speed-up vs Sparse", stats.gmean(df["Speed-up vs Sparse"]))
print("Percent Faster", len(df[df["Speed-up vs Sparse"] > 1]) / len(df))

for bcols in df["n"].unique():
    print("Bcols", bcols)
    df_bcols = df[df["n"] == bcols]
    print("    Speed-up vs Dense", stats.gmean(df_bcols["Speed-up vs Dense"]))
    print("    Percent Faster", len(df_bcols[df_bcols["Speed-up vs Dense"] > 1]) / len(df_bcols))
    print("    Speed-up vs Sparse", stats.gmean(df_bcols["Speed-up vs Sparse"]))
    print("    Percent Faster", len(df_bcols[df_bcols["Speed-up vs Sparse"] > 1]) / len(df_bcols))