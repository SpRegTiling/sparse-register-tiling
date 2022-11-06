import glob
import sys
import matplotlib.pyplot as plt
import numpy as np
from tools.paper.plotting.plot_utils import plot_save
from matplotlib import rc, rcParams
from tools.paper.plotting.plot_utils import *

SUBFOLDER = sys.argv[1] + '/'

NTHREADS = 16

df = load_dlmc_df(SUBFOLDER, nthreads=NTHREADS)

df = filter(df, best_nano=True)
print(df)
ax = df.plot.scatter(x='sparsity', y='Speed-up vs Dense', colormap='viridis', alpha=0.5, s=1)

print(list(df.columns))
print(df.sparsity)

plt.show(block=True)
