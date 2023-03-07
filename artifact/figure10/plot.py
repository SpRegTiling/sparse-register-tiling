import glob
import sys
import matplotlib.pyplot as plt
import numpy as np
from tools.paper.plotting.plot_utils import plot_save
from matplotlib import rc, rcParams
from tools.paper.plotting.plot_utils import *
from tools.paper.plotting import post_process

import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__)) + "/"

from artifact.utils import *

#
#   TODO: Fix papi issues in container
#

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 30})
plt.rcParams["figure.figsize"] = (20, 7)
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0
pd.options.display.max_columns = None
pd.options.display.max_rows = None

dft = pd.read_csv(RESULTS_DIR + "figure10_transformed.csv")
dfnt = pd.read_csv(RESULTS_DIR + "figure10_not_transformed.csv")

dft["datatransform"] = True
dfnt["datatransform"] = False

df = pd.concat([dft, dfnt])

nano_methods = df['name'].str.contains(r'M[0-9]N[0-9]', regex=True)
df["orig_name"] = df['name']
df.loc[nano_methods, 'name'] = "Sp. Reg."

df["include"] = (df["name"].str.contains("Sp. Reg.") | df["name"].str.contains("MKL_Dense"))
df = filter(df, include=True)

df["flops"] = 2 * df["n"] * df["nnz"]
df["gflops/s"] = (df["flops"] / (df["time median"]/1e6)) / 1e9

df.loc[df["datatransform"] == True, "name2"] = "transformed"
df.loc[df["datatransform"] == False, "name2"] = "not-transformed"
df.loc[df["name"].str.contains("MKL_Dense"), "name2"] = "dense"

df["sparsity"] = round(1 - df["nnz"] / (df["m"] * df["k"]), 2)
df["sparsity_raw"] = 1 - df["nnz"] / (df["m"] * df["k"])

bColsList = [32, 128, 256, 512]
numThreadsList = [1]
fig, axs = plt.subplots(len(numThreadsList), len(bColsList))
plt.locator_params(nbins=4)
dimw = 0.6
alpha = 1

adf = df[df['name2'] == 'transformed']['gflops/s'] - df[df['name2'] == 'not-transformed']['gflops/s']
adf = adf[adf.notna()]
a1 = (df[df['name2'] == 'transformed']['gflops/s'] - df[df['name2'] == 'not-transformed']['gflops/s']).mean()
a2 = (df[df['name2'] == 'not-transformed']['gflops/s'] - df[df['name2'] == 'dense']['gflops/s']).mean()

handles, labels = [], []

dftmp = df
merged_chart = None

for bcols in range(len(bColsList)):
    df_filtered = filter(dftmp, n=bColsList[bcols])
    df_filtered = df_filtered.sort_values(by=['sparsity'])
    x = np.arange(len(df_filtered[df_filtered['name2'] == 'transformed']['sparsity'])) + 1
    axs[bcols].bar(x, df_filtered[df_filtered['name2'] == 'transformed']['gflops/s'], dimw, color='royalblue', alpha=alpha, label='unroll-and-sparse-jam + data compression')
    axs[bcols].bar(x, df_filtered[df_filtered['name2'] == 'not-transformed']['gflops/s'], dimw, color='salmon', alpha=0.8, label='unroll-and-sparse-jam')
    axs[bcols].bar(x + dimw/2, df_filtered[df_filtered['name2'] == 'dense']['gflops/s'], dimw/2, color='green', alpha=alpha, label='MKL (sgemm)')
    if bcols == 0:
        handles.extend(axs[bcols].get_legend_handles_labels()[0])
        labels.extend(axs[bcols].get_legend_handles_labels()[1])
    axs[bcols].spines.right.set_visible(False)
    axs[bcols].spines.top.set_visible(False)
    axs[bcols].set_xticks(x)
    axs[bcols].set_xlabel('Matrix Instance')
    every_nth = 3
    for n, label in enumerate(axs[bcols].xaxis.get_ticklabels()):
        if n % every_nth != 0:
            label.set_visible(False)

axs[0].set_ylabel('Effective GFLOP/s')
plt.subplots_adjust(hspace=0.4, wspace=0.3)

fig.legend(handles, labels, loc='upper center', framealpha=0.3, ncol=len(handles))
filepath = PLOTS_DIR + 'figure10.pdf'
filepath = filepath.replace(".pdf", "") + ".jpg"
plt.margins(x=0)
plt.tight_layout(rect=(0,0,1,0.92))
plt.savefig(filepath)
print("Created:", filepath)