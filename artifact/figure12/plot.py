import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
import sys; sys.path.insert(0,f'{SCRIPT_DIR}/../')
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from pandas.api.types import is_string_dtype
import altair as alt
import altair_saver
from collections import defaultdict
from scipy.stats.mstats import gmean
import glob
import seaborn as sns

from artifact.utils import *
from artifact.figure7_to_9.post_process_results import get_df, thread_list, bcols_list

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 15})
plt.rcParams["figure.figsize"] = (12, 4)
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0

def create_chart_grid_and_save(charts, row_width):
    charts_merged = None
    charts_row = None

    col = 0
    for i in range(0, len(charts)):
        col = i % row_width
        if col == 0:
            charts_merged = charts_row if charts_merged is None else charts_merged & charts_row
            charts_row = None

        charts_row = charts[i] if charts_row is None else charts_row | charts[i]

    if col:
        charts_merged = charts_row if charts_merged is None else charts_merged & charts_row
    print("charts_merged", charts_merged)
    return charts_merged


alt.data_transformers.enable('default', max_rows=1000000)
unique_config_columns = ["name", "n", "m_tile", "k_tile", "n_tile"]

PLOT_GFLOPS = True
PLOT_SPEEDUP = True
ADD_REF_LINE = False
color_scheme = 'purpleblue'

df = pd.read_csv(RESULTS_DIR + f'/figure12.csv')

bcols_charts = defaultdict(lambda: [])
numThreads_charts = defaultdict(lambda: [])

def subtract_mkl_flops(df, name, group_by=["matrixPath", "n", "numThreads"]):
    def compute_time_vs(x):
        baseline = x[x["name"] == name].iloc[0]["SP_FLOPS_TOTAL"]
        x[f'SP_FLOPS_TOTAL-mkl'] = x["SP_FLOPS_TOTAL"] - baseline
        return x

    df = df.groupby(group_by).apply(compute_time_vs).reset_index(drop=True)
    return df

df = subtract_mkl_flops(df, 'MKL_Sparse')

df["sparsity"] = round(1 - df["nnz"] / (df["m"] * df["k"]), 2)
df["sparsity_raw"] = 1 - df["nnz"] / (df["m"] * df["k"])

b_colss = sorted(df["n"].unique())
n_threadss = df["numThreads"].unique()
methods = df["name"].unique()
n = 256

df["gflops"] = 2 * df["n"] * df["nnz"]
df["gflops/s"] = (df["gflops"] / df["time median"]) / 1e9

df["LOADS_AND_STORES"] = df["UOPS_LOADS"] + df["UOPS_STORES"]

methods = {
    'Best Nano': 'Best Nano-Kernel',
    'ASpT' : 'ASpT',
    'MKL_Dense': 'MKL Dense',
    'MKL_Sparse': 'MKL Sparse'
}

df["density"] = 1 - df["sparsity"]
df["time_ms"] = df["time median"] * 1000
density_thresh = 1.0
density_thresh_2 = 0.3

bColsList = [256]
fig, axs = plt.subplots(1, 3)
handles, labels = [], []
for n in range(len(bColsList)):
    df_filtered = df[(df["numThreads"] == 1) \
                    #  & (df["name"].isin(methods.keys())) \
                        & (df["density"] <= density_thresh) \
                        & (df["n"] == bColsList[n])]
    
    # df_filtered.loc[df_filtered['global_best'] == True, 'name'] = 'Best Nano'
    # df_filtered[df_filtered['name'] == 'M8N3_NKM_LB_TLB128_SA_orig'] = 'Best Nano'
    df_filtered.sort_values(by=['sparsity'], inplace=True)


    axs[0].plot(df_filtered[df_filtered['name'] == 'M8N3_NKM_LB_TLB128_SA_orig']['sparsity'], df_filtered[df_filtered['name'] == 'M8N3_NKM_LB_TLB128_SA_orig']['time median'], alpha=0.7, label='Sparse Reg Tiling')
    axs[0].plot(df_filtered[df_filtered['name'] == 'ASpT']['sparsity'], df_filtered[df_filtered['name'] == 'ASpT']['time median'], alpha=0.7, label='ASpT')
    axs[0].plot(df_filtered[df_filtered['name'] == 'MKL_Dense']['sparsity'], df_filtered[df_filtered['name'] == 'MKL_Dense']['time median'], alpha=0.7, label='MKL Dense')
    axs[0].plot(df_filtered[df_filtered['name'] == 'MKL_Sparse']['sparsity'], df_filtered[df_filtered['name'] == 'MKL_Sparse']['time median'], alpha=0.7, label='MKL Sparse')
    axs[0].set_ylabel(f'Execution Time')
    # axs[0].set_title(f'B Columns={bColsList[n]}')
    axs[0].spines.right.set_visible(False)
    axs[0].spines.top.set_visible(False)
    if n == 0:
        handles.extend(axs[0].get_legend_handles_labels()[0])
        labels.extend(axs[0].get_legend_handles_labels()[1])

    axs[1].plot(df_filtered[df_filtered['name'] == 'M8N3_NKM_LB_TLB128_SA_orig']['sparsity'], df_filtered[df_filtered['name'] == 'M8N3_NKM_LB_TLB128_SA_orig']['UOPS_LOADS']/1e6, alpha=0.7, label='Sparse Reg Tiling')
    axs[1].plot(df_filtered[df_filtered['name'] == 'ASpT']['sparsity'], df_filtered[df_filtered['name'] == 'ASpT']['UOPS_LOADS']/1e6, alpha=0.7, label='ASpT')
    axs[1].plot(df_filtered[df_filtered['name'] == 'MKL_Dense']['sparsity'], df_filtered[df_filtered['name'] == 'MKL_Dense']['UOPS_LOADS']/1e6, alpha=0.7, label='MKL Dense')
    axs[1].plot(df_filtered[df_filtered['name'] == 'MKL_Sparse']['sparsity'], df_filtered[df_filtered['name'] == 'MKL_Sparse']['UOPS_LOADS']/1e6, alpha=0.7, label='MKL Sparse')
    axs[1].set_ylabel(f'Loads')
    axs[1].spines.right.set_visible(False)
    axs[1].spines.top.set_visible(False)
    # handles, labels = axs[1].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper center', framealpha=0.3)

    axs[2].plot(df_filtered[df_filtered['name'] == 'M8N3_NKM_LB_TLB128_SA_orig']['sparsity'], df_filtered[df_filtered['name'] == 'M8N3_NKM_LB_TLB128_SA_orig']['SP_FLOPS_TOTAL-mkl']/1e6, alpha=0.7, label='Sparse Reg Tiling')
    axs[2].plot(df_filtered[df_filtered['name'] == 'ASpT']['sparsity'], df_filtered[df_filtered['name'] == 'ASpT']['SP_FLOPS_TOTAL-mkl']/1e6, alpha=0.7, label='ASpT')
    axs[2].plot(df_filtered[df_filtered['name'] == 'MKL_Dense']['sparsity'], df_filtered[df_filtered['name'] == 'MKL_Dense']['SP_FLOPS_TOTAL-mkl']/1e6, alpha=0.7, label='MKL Dense')
    axs[2].plot(df_filtered[df_filtered['name'] == 'MKL_Sparse']['sparsity'], df_filtered[df_filtered['name'] == 'MKL_Sparse']['SP_FLOPS_TOTAL-mkl']/1e6, alpha=0.7, label='MKL Sparse')
    axs[2].set_ylabel(f'Redundant GFLOPs')
    axs[2].spines.right.set_visible(False)
    axs[2].spines.top.set_visible(False)
    # handles, labels = axs[2].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper center', framealpha=0.3)

axs[0].set_xlabel('Sparsity')
axs[1].set_xlabel('Sparsity')
axs[2].set_xlabel('Sparsity')

plt.subplots_adjust(hspace=0.3, wspace=0.3)
fig.legend(handles, labels, loc='upper center', ncol=len(handles))

filepath = PLOTS_DIR + 'figure12.jpg'
plt.margins(x=0)
plt.tight_layout(rect=(0,0,1,0.9))
plt.savefig(filepath)
