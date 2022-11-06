import glob
import sys
import matplotlib.pyplot as plt
import numpy as np
from tools.paper.plotting.plot_utils import plot_save
from matplotlib import rc, rcParams
from tools.paper.plotting.plot_utils import *

SUBFOLDER = sys.argv[1] + '/'
NTHREADS = {
    'cascadelake/': 16,
    'raspberrypi/': 4,
}[SUBFOLDER]

df = load_dlmc_df(SUBFOLDER, nthreads=NTHREADS)

df = filter(df, best_nano=True)
df = df[(df["sparsity"] <= 0.95) & (df["sparsity"] >= 0.7) & (df["n"] < 1024)]

ax = df.plot.scatter(x='gflops', y='gflops/s', c='sparsity', colormap='cividis', alpha=0.5, s=1)
ax.set_xscale('log')
ax.axhline(y=1.0, color='r', linestyle='-')

plt.ylabel('Speed-up vs MKL Sparse')
plt.xlabel('Problem Size (GFLOPs)')
plot_save(f"scatters/{SUBFOLDER}/vs_sparse_gflopss")

ax = df.plot.scatter(x='gflops', y='gflops/s', c='sparsity', colormap='cividis', alpha=0.5, s=1)
ax.set_xscale('log')
ax.axhline(y=1.0, color='r', linestyle='-')

plt.ylabel('Speed-up vs MKL Sparse')
plt.xlabel('Problem Size (GFLOPs)')
plot_save(f"scatters/{SUBFOLDER}/vs_dense_gflopss")
