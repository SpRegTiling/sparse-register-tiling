import glob
import matplotlib.pyplot as plt
import numpy as np
from tools.paper.plotting.plot_utils import plot_save
from matplotlib import rc, rcParams

# activate latex text rendering
# rc('text', usetex=True)
# rc('axes', linewidth=2)
# rc('font', weight='bold')
# rcParams['text.latex.preamble'] = [r'\usepackage{sfmath} \boldmath']

DLMC_DATA = "/sdb/cache/workingset_size/dlmc"
SS_DATA = "/sdb/cache/workingset_size/ss"

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 14})
plt.rcParams["figure.figsize"] = (7, 6)
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0



SS_SPARSITY_RANGES = [0.7, 1.0]
DLMC_SPARSITY_RANGES = [0.7, 1.0]

np.random.seed(42)

fig, ax = plt.subplots()
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)


def rand_jitter(arr):
    stdev = .005 * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev


tile_sizes = np.load(f"{DLMC_DATA}/tile_sizes.npy")
dlmc_files = np.random.choice(glob.glob(DLMC_DATA + "/**/*_working_set.npz", recursive=True), 200)

tile_sizes = np.load(f"{SS_DATA}/tile_sizes.npy")
ss_files = np.random.choice(glob.glob(SS_DATA + "/**/*_working_set.npz", recursive=True), 200)

for file in ss_files:
    working_set_sizes = np.load(file)
    cov = working_set_sizes["cov"]
    if len(cov) == 0: continue
    sparsity = working_set_sizes["sparsity"]

    if sparsity < SS_SPARSITY_RANGES[0] or sparsity > SS_SPARSITY_RANGES[1]: continue
    last_ss = plt.scatter(rand_jitter(tile_sizes[:len(cov)]), cov, color="navy", label="SuiteSparse",
                          alpha=float(sparsity), s=1)

for file in dlmc_files:
    working_set_sizes = np.load(file)
    cov = working_set_sizes["cov"]
    sparsity = working_set_sizes["sparsity"]

    if sparsity < DLMC_SPARSITY_RANGES[0] or sparsity > DLMC_SPARSITY_RANGES[1]: continue
    last_dlmc = plt.scatter(rand_jitter(tile_sizes[:len(cov)]), cov, color="orangered", label="DLMC",
                            alpha=float(sparsity), s=1)


ticks = [f'{i}x{i}' for i in tile_sizes]
plt.xticks(tile_sizes[::6], ticks[::6], rotation=45)
plt.xlabel(r"Tile Size")
plt.ylabel(r"Coefficient of Variation")

#plt.tight_layout()

lgnd = plt.legend(handles=[last_ss, last_dlmc], loc='upper right')
lgnd.legendHandles[0]._sizes = [10]
lgnd.legendHandles[1]._sizes = [10]
#
plt.margins(x=0)
plt.tight_layout()

plot_save("working_set_size")

