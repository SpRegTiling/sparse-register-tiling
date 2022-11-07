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
    'raspberrypi/': 1,
}[SUBFOLDER]

df = load_dlmc_df(SUBFOLDER, nthreads=NTHREADS)
df = filter(df, best_nano=True)
df = df[(df["sparsity"] <= 0.95) & (df["sparsity"] >= 0.6) & (df["n"] < 1024)]
df["m_tile"] = df["config"].str.extract("m_tile:(\d+)")
df["k_tile"] = df["config"].str.extract("k_tile:(\d+)")

df = df[df['m_tile'].notna()]
df = df[df['k_tile'].notna()]

df["m_tile"] = df["m_tile"].astype(int)
df["k_tile"] = df["k_tile"].astype(int)

df["m_tiles"] = np.ceil(df['m'] / df["m_tile"].astype(int))
df["k_tiles"] = np.ceil(df['k'] / df["k_tile"].astype(int))

df["num_panels"] = df["m_tiles"] * df["k_tiles"]

print(df["config"].unique())
print(df["m_tile"].unique())
print(df["k_tile"].unique())

#
#   Correct for miscalulation of storage when recording, recorded floats and ints as 8 bytes (size of ptr)
#     recorded extra overhead that is not actually needed (df["num_panels"] * 3 * 64) for alignment
#     and  df["num_panels"] * 40 for legacy flags, add back 8 bytes per panel which is realistically needed
#

df['required_storage_pct'] = \
    ((df['required_storage'] - (df["num_panels"] * 3 * 64) - df["num_panels"] * 40 + df["num_panels"]) / 2 \
    + df["num_panels"] * 2* 4) \
    / (df['m'] * df['k'] * 4)

df['csr_required_storage_pct'] = \
    (df["nnz"] * 4 * 2 + (df["m"] + 1) * 4) \
    / (df['m'] * df['k'] * 4)


print("pct_lower", len(df[df["required_storage_pct"] < df["csr_required_storage_pct"]]) / len(df))

plt.scatter(x=rand_jitter(df["sparsity_raw"]), y=df["required_storage_pct"], alpha=0.5, s=1)
plt.scatter(x=rand_jitter(df["sparsity_raw"]), y=df["csr_required_storage_pct"], alpha=0.5, s=1, color='red')

plt.ylabel('Required Storage (pct of dense)')
plt.xlabel('Sparsity')
plot_save(f"scatters/{SUBFOLDER}/required_storage_pct")
