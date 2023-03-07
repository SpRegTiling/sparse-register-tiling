import glob
import sys
import matplotlib.pyplot as plt
import numpy as np
from tools.paper.plotting.plot_utils import plot_save
from matplotlib import rc, rcParams
from tools.paper.plotting.plot_utils import *

from artifact.utils import *

#
#   TODO: Fix papi issues in container
#

df["loads_per_fma"] = df["UOPS_LOADS"] / ((df["SP_AVXW"] + df["SP_AVX"] + df["SP_SSE"] + df["SP_SINGLE"]) / 2)
df = df[df['loads_per_fma'] < 5]
df = df[df['loads_per_fma'].isna() == False]


ax = df.plot.scatter(x='loads_per_fma', y='Speed-up vs Sparse', c='sparsity', colormap='cividis', alpha=0.5, s=1)
ax.axhline(y=1.0, color='r', linestyle='-')

plt.ylabel('Speed-up vs MKL Sparse')
plt.xlabel('Average Loads per Floating Multiply-Add Operations')
plt.savefig(PLOTS_DIR + "/figure11.jpg")
