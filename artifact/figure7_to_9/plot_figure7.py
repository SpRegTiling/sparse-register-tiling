import glob
import sys
import matplotlib.pyplot as plt
import numpy as np
from tools.paper.plotting.plot_utils import plot_save
from matplotlib import rc, rcParams
from tools.paper.plotting.plot_utils import *
from scipy import stats

from artifact.utils import *
from artifact.figure7_to_9.post_process_results import get_df, thread_list, bcols_list

chipset = 'cascadelake/'
NTHREADS = {
    'cascadelake/': 20,
    'raspberrypi/': 4,
}[chipset]

BASLINE_SPARSE = {
    'cascadelake/': "MKL (spmm, csr)",
    'raspberrypi/': "XNN (spmm, 16x1)",
}[chipset]

BASLINE_DENSE = {
    'cascadelake/': "MKL (sgemm)",
    'raspberrypi/': "ARMCL (sgemm)",
}[chipset]

dfs = []
for bcol in bcols_list():
    dfs.append(get_df(bcol, NTHREADS))
df = pd.concat(dfs)

df["Speed-up vs Dense"] = df[f"time cpu median|MKL_Dense"] / df[f"time cpu median|Sp. Reg."]
df["Speed-up vs Sparse"] = df[f"time cpu median|MKL_Sparse"] / df[f"time cpu median|Sp. Reg."]

df = df[df["Speed-up vs Dense"] > 0.0]
df = df[df["Speed-up vs Dense"] < 80]

ax = df.plot.scatter(x='gflops', y='Speed-up vs Sparse', c='sparsity', colormap='cividis', alpha=0.5, s=1)
ax.set_xscale('log')
ax.axhline(y=1.0, color='r', linestyle='-')
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)

plt.ylabel(f'Speed-up vs {BASLINE_SPARSE}')
plt.xlabel('Problem Size (GFLOPs)')
plt.margins(x=0)
plt.tight_layout()
plt.savefig(PLOTS_DIR + f"/figure7_left_side.jpg")

plt.clf()
plt.close()
plt.cla()

ax = df.plot.scatter(x='gflops', y='Speed-up vs Dense', c='sparsity', colormap='cividis', alpha=0.5, s=1)
ax.set_xscale('log')
ax.axhline(y=1.0, color='r', linestyle='-')
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)

plt.ylabel(f'Speed-up vs {BASLINE_DENSE}')
plt.xlabel('Problem Size (GFLOPs)')
plt.margins(x=0)
plt.tight_layout()
f = plt.gcf()
cax = f.get_axes()[1].remove()
plt.savefig(PLOTS_DIR + f"/figure7_right_side.jpg")