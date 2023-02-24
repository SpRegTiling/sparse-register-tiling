import glob
import sys
import math
import matplotlib.pyplot as plt
import numpy as np
from tools.paper.plotting.plot_utils import plot_save
from matplotlib import rc, rcParams
from tools.paper.plotting.plot_utils import *
from scipy import stats
from tabulate import tabulate

SUBFOLDER = sys.argv[1] + '/'
NTHREADS = {
    'cascadelake/': 16,
    'raspberrypi/': 4,
}[SUBFOLDER]

BASLINE = {
    'cascadelake/': 16,
    'raspberrypi/': 4,
}[SUBFOLDER]

df = load_dlmc_df(SUBFOLDER, nthreads=16)
df = df[(df["sparsity"] <= 0.95) & (df["sparsity"] >= 0.6) & (df["n"] < 1024)]
df = df.reset_index(drop=True)

matrices = list(df["matrixPath"].unique())
matrices = ['dlmc' + x.split('dlmc')[-1] for x in matrices]

per_part = int(math.ceil(len(matrices) / 5))

test = []
part = 1
for i in range(0, len(matrices), per_part):
    with open(f'/sdb/codegen/spmm-nano-bench/tools/filelists/dlmc_part{part}.txt', 'w+') as f:
      f.write("\n".join(matrices[i:i+per_part]))
    part += 1
