import glob
import sys
import matplotlib.pyplot as plt
import numpy as np
from tools.paper.plotting.plot_utils import plot_save
from matplotlib import rc, rcParams
from tools.paper.plotting.plot_utils import *

from artifact.utils import *
from artifact.figure7_to_9.post_process_results import get_df, thread_list, bcols_list, post_process

chipeset = 'cascadelake/'
NTHREADS = {
    'cascadelake/': 1,
    'raspberrypi/': 1,
}[chipeset]

files = glob.glob(RESULTS_DIR + "/figure7_to_9_results*.csv")
dfs = []
for file in files:
    dfs.append(pd.read_csv(file))

df = pd.concat(dfs)
df = post_process(df)
df = filter(df, is_nano=True)

df["m_tile"] = df["config"].str.extract("m_tile:(\d+)")
df["k_tile"] = df["config"].str.extract("k_tile:(\d+)")

df = df[df['m_tile'].notna()]
df = df[df['k_tile'].notna()]

df["m_tile"] = df["m_tile"].astype(int)
df["k_tile"] = df["k_tile"].astype(int)

df["m_tiles"] = np.ceil(df['m'] / df["m_tile"].astype(int))
df["k_tiles"] = np.ceil(df['k'] / df["k_tile"].astype(int))

df["num_panels"] = df["m_tiles"] * df["k_tiles"]
# df["sparsity_raw"] = 1 - df["nnz"] / (df["m"] * df["k"])

print(df["config"].unique())
print(df["m_tile"].unique())
print(df["k_tile"].unique())

#
#   Correct for miscalculation of storage when recording, recorded floats and ints as 8 bytes (size of ptr)
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

fig, ax = plt.subplots()
sc1 = plt.scatter(x=rand_jitter(df["sparsity_raw"]), y=df["required_storage_pct"], alpha=0.5, color='deepskyblue', s=1, label="Sparse Reg Tiling")
sc2 = plt.scatter(x=rand_jitter(df["sparsity_raw"]), y=df["csr_required_storage_pct"], alpha=0.5, color='firebrick', s=1, label='CSR')
lgnd = plt.legend(handles=[sc1, sc2], loc='upper right')
lgnd.legendHandles[0]._sizes = [50]
lgnd.legendHandles[1]._sizes = [50]
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)

plt.ylabel('Required Storage (pct of dense)')
plt.xlabel('Sparsity')
plt.margins(x=0)
plt.tight_layout()
plt.savefig(PLOTS_DIR + f"/figure9.jpg")