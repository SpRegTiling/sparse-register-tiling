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

df = pd.read_csv(RESULTS_DIR + '/cache/cascadelake/part1_per_part.csv')
# df16 = load_dlmc_df('cascadelake/', nthreads=16)
# df20 = load_dlmc_df('cascadelake/', nthreads=20)

df16 = filter(df, best_nano=True, numThreads=16)
df20 = filter(df, best_nano=True, numThreads=20)

df16 = df16[(df16["sparsity"] <= 0.95) & (df16["sparsity"] >= 0.7) & (df16["n"] < 1024)]
df20 = df20[(df20["sparsity"] <= 0.95) & (df20["sparsity"] >= 0.7) & (df20["n"] < 1024)]

print("16 Vs Dense", stats.gmean(df16["Speed-up vs Dense"]))
print("20 Vs Dense", stats.gmean(df20["Speed-up vs Dense"]))
print("16 Vs Sparse", stats.gmean(df16["Speed-up vs Sparse"]))
print("20 Vs Sparse", stats.gmean(df20["Speed-up vs Sparse"]))

ax = df16.plot.scatter(x='gflops', y='Speed-up vs Dense', c='sparsity', colormap='cividis', alpha=0.5, s=1)
ax.set_xscale('log')
ax.axhline(y=1.0, color='r', linestyle='-')

plt.ylabel('Speed-up vs MKL Dense')
plt.xlabel('Problem Size (GFLOPs)')
plot_save(f"scatters/{SUBFOLDER}/vs_dense_16")

ax = df20.plot.scatter(x='gflops', y='Speed-up vs Dense', c='sparsity', colormap='cividis', alpha=0.5, s=1)
ax.set_xscale('log')
ax.axhline(y=1.0, color='r', linestyle='-')

plt.ylabel('Speed-up vs MKL Dense')
plt.xlabel('Problem Size (GFLOPs)')
plot_save(f"scatters/{SUBFOLDER}/vs_dense_20")

ax = df16.plot.scatter(x='gflops', y='Speed-up vs Sparse', c='sparsity', colormap='cividis', alpha=0.5, s=1)
ax.set_xscale('log')
ax.axhline(y=1.0, color='r', linestyle='-')

plt.ylabel('Speed-up vs MKL Sparse')
plt.xlabel('Problem Size (GFLOPs)')
plot_save(f"scatters/{SUBFOLDER}/vs_sparse_16")

ax = df20.plot.scatter(x='gflops', y='Speed-up vs Sparse', c='sparsity', colormap='cividis', alpha=0.5, s=1)
ax.set_xscale('log')
ax.axhline(y=1.0, color='r', linestyle='-')

plt.ylabel('Speed-up vs MKL Sparse')
plt.xlabel('Problem Size (GFLOPs)')
plot_save(f"scatters/{SUBFOLDER}/vs_sparse_20")