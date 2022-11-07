import glob
import sys
import matplotlib.pyplot as plt
import numpy as np
from tools.paper.plotting.plot_utils import plot_save
from matplotlib import rc, rcParams
from tools.paper.plotting.plot_utils import *

SUBFOLDER = sys.argv[1] + '/'
NTHREADS = {
    'cascadelake/': 1,
    'raspberrypi/': 4,
}[SUBFOLDER]

df = load_dlmc_df(SUBFOLDER, nthreads=NTHREADS)

print(df['gflops'].min())

# df = filter(df, best_nano=True)
df = df[df['best_nano'] == True]
df = df[(df["sparsity"] <= 0.95) & (df["sparsity"] >= 0.6) & (df["n"] < 1024)]

print(df)
df = df[df['loads_per_fma'].isna() == False]
print(df['loads_per_fma'].unique())

# ax = df.plot.scatter(x='gflops', y='Speed-up vs Sparse', c='sparsity', colormap='cividis', alpha=0.5, s=1)
# ax.set_xscale('log')
# ax.axhline(y=1.0, color='r', linestyle='-')
#
# plt.ylabel('Speed-up vs MKL Sparse')
# plt.xlabel('Problem Size (GFLOPs)')
# plot_save(f"scatters/{SUBFOLDER}/vs_sparse")

# plt.scatter(x=rand_jitter(df["loads_per_fma"]), y=df["Speed-up vs Sparse"], alpha=0.5, s=1)
# plt.gca().axhline(y=1.0, color='r', linestyle='-')

# plt.ylabel('Speed-up vs MKL Sparse')
# plt.xlabel('Loads per FMA')
# plot_save(f"scatters/{SUBFOLDER}/vs_loads_per_fma")
df["loads_per_fma"] = df["UOPS_LOADS"] / ((df["SP_AVXW"] + df["SP_AVX"] + df["SP_SSE"] + df["SP_SINGLE"]) / 2)
df = df[df['loads_per_fma'] < 5]


ax = df.plot.scatter(x='loads_per_fma', y='Speed-up vs Sparse', c='sparsity', colormap='cividis', alpha=0.5, s=1)
# ax.set_xscale('log')
ax.axhline(y=1.0, color='r', linestyle='-')

plt.ylabel('Speed-up vs MKL Sparse')
plt.xlabel('Average Loads per Floating Multiply-Add Operations')
plot_save(f"scatters/{SUBFOLDER}/vs_loads_per_fma")
