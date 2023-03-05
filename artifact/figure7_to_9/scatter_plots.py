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
    'cascadelake/': "MKL (spmm, csr)",
    'raspberrypi/': "XNN (spmm, 16x1)",
}[SUBFOLDER]

df = load_dlmc_df(SUBFOLDER, nthreads=NTHREADS)

df = filter(df, best_nano=True)
df = df[(df["sparsity"] <= 0.95) & (df["sparsity"] >= 0.6) & (df["n"] < 1024)]
df = df[df["Speed-up vs Dense"] > 0.0]
df = df[df["Speed-up vs Dense"] < 200]

ax = df.plot.scatter(x='gflops', y='Speed-up vs Sparse', c='sparsity', colormap='cividis', alpha=0.5, s=1)
ax.set_xscale('log')
ax.axhline(y=1.0, color='r', linestyle='-')

plt.ylabel(f'Speed-up vs {BASLINE}')
plt.xlabel('Problem Size (GFLOPs)')
plot_save(f"scatters/{SUBFOLDER}/vs_sparse_{SUBFOLDER.strip('/')}")

plt.clf()
plt.close()
plt.cla()

plt.scatter(x=rand_jitter(df["sparsity_raw"]), y=df["Speed-up vs Sparse"], alpha=0.5, s=1)
plt.gca().axhline(y=1.0, color='r', linestyle='-')

plt.ylabel(f'Speed-up vs {BASLINE}')
plt.xlabel('Problem Size (GFLOPs)')
plot_save(f"scatters/{SUBFOLDER}/vs_sparse_jitter_{SUBFOLDER.strip('/')}")


ax = df.plot.scatter(x='gflops', y='Speed-up vs Dense', c='sparsity', colormap='cividis', alpha=0.5, s=1)
ax.set_xscale('log')
ax.axhline(y=1.0, color='r', linestyle='-')

plt.ylabel(f'Speed-up vs {BASLINE}')
plt.xlabel('Problem Size (GFLOPs)')
plot_save(f"scatters/{SUBFOLDER}/vs_dense_{SUBFOLDER.strip('/')}")