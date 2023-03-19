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
import matplotlib.ticker as mtick   

from tools.paper.plotting.post_process import compute_speed_up_vs, compute_matrix_properties, compute_best, compute_global_best
from tools.paperv2.dlmc.utils import *

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

cluster_info = {
    "graham": ("~/graham/", 256),
    "niagara": ("~/niagara/", 512),
}

cluster = "niagara"
cluster_dir, vec_width = cluster_info[cluster]

all_files = glob.glob(os.path.join('/sdb/paperv2_results/random/', "*.csv" ))
df = pd.concat((pd.read_csv(f) for f in all_files), axis=0, ignore_index=True)

bcols_charts = defaultdict(lambda: [])
numThreads_charts = defaultdict(lambda: [])

#df = compute_speed_up_vs(df, 'MKL_Dense', group_by=["matrixPath", "n", "numThreads"])
df = compute_matrix_properties(df, sparsity_round=False)
#df = compute_best(df)
#df = compute_global_best(df)

df.to_csv('/sdb/paperv2_results/cache/random_sweep_cache.csv')
df = pd.read_csv('/sdb/paperv2_results/cache/random_sweep_cache.csv')

def subtract_mkl_flops(df, name, group_by=["matrixPath", "n", "numThreads"]):
    def compute_time_vs(x):
        baseline = x[x["name"] == name].iloc[0]["SP_FLOPS_TOTAL"]
        x[f'SP_FLOPS_TOTAL-mkl'] = x["SP_FLOPS_TOTAL"] - baseline
        return x

    df = df.groupby(group_by).apply(compute_time_vs).reset_index(drop=True)
    return df

df = subtract_mkl_flops(df, 'MKL_Sparse')

b_colss = sorted(df["n"].unique())
n_threadss = df["numThreads"].unique()
methods = df["name"].unique()
n = 256

df["gflops"] = 2 * df["n"] * df["nnz"]
df["gflops/s"] = (df["gflops"] / df["time median"]) / 1e9

df["LOADS_AND_STORES"] = df["UOPS_LOADS"] + df["UOPS_STORES"]

print(df["name"].unique())


methods = {
    'Best Nano': 'Best Nano-Kernel',
    'ASpT' : 'ASpT',
    'MKL_Dense': 'MKL Dense',
    'MKL_Sparse': 'MKL Sparse'
}

df["density"] = 1 - df["sparsity"]
df["time_ms"] = df["time median"] * 1000
df["loads_per_fma"] = df["UOPS_LOADS"] / ((df["SP_AVXW"] + df["SP_AVX"] + df["SP_SSE"] + df["SP_SINGLE"]) / 2)
df.loc[df['name'] == 'M8N3_NKM_LB_TLB128_SA_orig', "name"] = "Sp. Reg."
df.loc[df['name'] == 'ASpT', "name"] = "ASpT Best" # single threaded no-need to check the increased parallelism version

density_thresh = 1.0
density_thresh_2 = 0.3
alpha = 0.9

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


    idx = 0
    ax = axs[idx]
    for method, color, label in intel_mcl:
        ax.plot(df_filtered[df_filtered['name'] == method]['sparsity']*100, df_filtered[df_filtered['name'] == method]['time median']/1e3, alpha=alpha, c=color, label=label)
    ax.set_ylabel(f'Execution Time (ms)')
    # ax.set_title(f'B Columns={bColsList[n]}')
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    if n == 0:
        handles.extend(ax.get_legend_handles_labels()[0])
        labels.extend(ax.get_legend_handles_labels()[1])

    idx += 1
    ax = axs[idx]
    for method, color, label in intel_mcl:
        ax.plot(df_filtered[df_filtered['name'] == method]['sparsity']*100, df_filtered[df_filtered['name'] == method]['SP_FLOPS_TOTAL-mkl']/1e6, alpha=alpha, c=color, label=label)
    ax.set_ylabel(f'Redundant GFLOPs')
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    # handles, labels = axs[2].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper center', framealpha=0.3)

    idx += 1
    ax = axs[idx]
    for method, color, label in intel_mcl:
        ax.plot(df_filtered[df_filtered['name'] == method]['sparsity']*100, df_filtered[df_filtered['name'] == method]['loads_per_fma'], alpha=alpha, c=color, label=label)
    ax.set_ylabel(f'Loads-per-FMA')
    ax.set_ylim(0, 2.8)

    
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)


for ax in axs:
    ax.axvline(x = 95, color = 'black', linewidth=0.5, linestyle=(0, (3, 5)), alpha=0.7)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_xlabel('Sparsity')
    lim = ax.get_xlim()
    ax.set_xticks(list(ax.get_xticks()) + [95])
    ax.set_xlim(lim)

plt.subplots_adjust(hspace=0.3, wspace=0.3)
fig.legend(handles, labels, loc='upper center', ncol=len(handles))

plt.margins(x=0)
plt.tight_layout(rect=(0,0,1,0.9))
savefig('random.pdf')