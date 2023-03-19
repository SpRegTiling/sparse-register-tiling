import glob
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns

from tools.paper.plotting.plot_utils import plot_save
from matplotlib import rc, rcParams
from tools.paper.plotting.plot_utils import *
from numpy import mean
from scipy.stats import gmean

from tools.paperv2.utils import *
from tools.paperv2.suitesparse_dlmc.post_process_results import ARCH, OUT_DIR

from matplotlib.ticker import StrMethodFormatter, NullFormatter

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 15})
plt.rcParams["figure.figsize"] = (3, 4)
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0

def speedup_column(method, baseline):
    return f'Speed-up {method} vs. {baseline}'

def compute_speedup(df, method, baseline):
    df[speedup_column(method, baseline)] = df[f"time median|{baseline}"] / df[f"time median|{method}"]

MARKSIZE = 2.5
MARKSIZE_DL = 3

colors = {
    32: 'DarkBlue',
    128: 'DarkRed'
}

THREAD_COUNT=20
BCOLS=128

SS_MARKER=">"
DL_MARKER="."

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 15})
plt.rcParams["figure.figsize"] = (6, 4)
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0

fig, axs = plt.subplots(1, 3, figsize=(16,5.5), squeeze=False)

##
#   All DLMC
##

ax = axs[0, 0]

tmp_files = glob.glob(f"{OUT_DIR}/figure_double*.bcols{BCOLS}.threads{THREAD_COUNT}")
dfs = []
for file in tmp_files:
    dfs.append(pd.read_csv(file))
dl_df = pd.concat(dfs)

tmp_files = glob.glob(f"{OUT_DIR}/psc_dlmc_{ARCH}*.bcols{BCOLS}.threads{THREAD_COUNT}")
dfs = []
for file in tmp_files:
    dfs.append(pd.read_csv(file))
psc_df = pd.concat(dfs)

# Manual horizontal merge
for val in ["gflops/s", "time median", "correct", "required_storage"]:
    dl_df[f"{val}|PSC"] = psc_df[f"{val}|PSC"]
df = dl_df
all_dlmc_df = df # save for stats

handles = []
labels = []

chipset = "cascade"
x_labels = ['60%-69.9%', '70%-79.9%', '80%-89.9%', '90%-95%']
x_ticks = [i+1 for i in range(len(x_labels))]

def _mean(list_in):
    geo_mean = []
    for sub_list in list_in:
        geo_mean.append(mean(sub_list))
    
    return geo_mean

mcl = intel_mcl_double
limits = (0, 500)

box_width = 0.15

def plot(ax, color, bias, data, label='nn'):
    geo_mean = _mean(data)
    ax.plot([x + bias - box_width * len(mcl)/2 for x in x_ticks], geo_mean, color=color, linewidth=1)
    return ax.boxplot(data, positions=[x + bias - box_width * len(mcl)/2 for x in x_ticks],
        notch=True, patch_artist=True,
        boxprops=dict(facecolor=color),
        capprops=dict(color=color),
        whiskerprops=dict(color=color),
        flierprops=dict(color=color, markeredgecolor=color, marker='o', markersize=0.5),
        medianprops=dict(color='black'),
        showfliers=True,
        #whis=(10,90),
        widths=0.15)

# for method, _, _ in mcl:
#     df[f'Speed-up {method} vs. {baseline}'] = df[f"time cpu median|{baseline}"] / df[f"time cpu median|{method}"]

# axs[i, j].plot([0.5, len(x_labels)+0.5],[1, 1], color='purple')
plots = []
labels = []

for mi, (method, color, label) in enumerate(mcl):
    data = [df[(df['sparsity']>=spBucket*0.1+0.6)&(df['sparsity']<(spBucket+1)*0.1+0.6)
                &(df[f'correct|{method}'] == "correct")
                &(~df[f'gflops/s|{method}'].isna())
                ][f'gflops/s|{method}'].tolist() for spBucket in range(len(x_labels))]
    plots.append(plot(ax, color, box_width*mi, data))
    labels.append(label)

handles = [plot["boxes"][0] for plot in plots]

ax.set_xticks(x_ticks)
ax.set_xticklabels(x_labels, rotation=22)
ax.set_xlim([0.5, len(x_labels)+0.5])
ax.set_xlabel('Sparsity')
ax.set_ylabel(f'Required GFLOP/s')
ax.set_title(f'All DLMC (60%-95%)', pad=10)
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)    
ax.set_ylim(limits)
ax.text(0.3, -0.27, "(a)", transform=ax.transAxes, size=18)

fig.legend(handles, labels, loc='upper center', ncol=len(handles))

##
#   500 of Each
##

#
#   Collect SS CSVs
#

ss_df = pd.read_csv(RESULTS_DIR + "/cachev2/suitesparse_dlmc/ss_with_cov.csv")
ss_df["cov|Sp. Reg."] = ss_df["cov"]

tmp_files = glob.glob(f"{OUT_DIR}/psc_ss_{ARCH}*.bcols{BCOLS}.threads{THREAD_COUNT}")
dfs = []
for file in tmp_files:
    dfs.append(pd.read_csv(file))
psc_ss_df = pd.concat(dfs)

# Manual horizontal merge
for val in ["gflops/s", "time median", "time cpu median", "correct", "required_storage"]:
    ss_df[f"{val}|PSC"] = psc_ss_df[f"{val}|PSC"]

#
#   Collect DLMC CSVs
#

dl_df = pd.read_csv(RESULTS_DIR + "/cachev2/suitesparse_dlmc/dlmc_with_cov.csv")

tmp_files = glob.glob(f"{OUT_DIR}/psc_dlmc_{ARCH}*.bcols{BCOLS}.threads{THREAD_COUNT}")
dfs = []
for file in tmp_files:
    dfs.append(pd.read_csv(file))
psc_dlmc_df = pd.concat(dfs)

for val in ["gflops/s", "time median", "time cpu median", "correct", "required_storage"]:
    dl_df[f"{val}|PSC"] = psc_dlmc_df[f"{val}|PSC"]

def filer_only_all_correct(df):
    bool_index = None
    for method in ["PSC", "Sp. Reg."]:
        if bool_index is None:
            bool_index = (df[f"correct|{method}"] == "correct")
        else:
            bool_index = bool_index & (df[f"correct|{method}"] == "correct")
    return df[bool_index]

dl_df = filer_only_all_correct(dl_df)
ss_df = filer_only_all_correct(ss_df)

#
#   Merge 
#

ss_df["ss"] = True
dl_df["ss"] = False

df = pd.concat((ss_df, dl_df))
print(len(df))

#
#   Plot
#

mcl = mcl[2:]

ax = axs[0, 1]
df["sparsity_raw"] = df["sparsity_raw"]
df["density"] = 1 - df["sparsity_raw"] 

ss = df["ss"]
df.loc[~ss & (df["density"] < 0.05), "density"] = 0.05 # normally we round so visually correct so minor outliers on the vertical line
ax = df[~ss].plot(kind='scatter', x="density", y='gflops/s|Sp. Reg.', c=SP_REG_COLOR, ax=ax, s=MARKSIZE_DL, marker=DL_MARKER)
ax = df[ss].plot(kind='scatter', x="density", y='gflops/s|Sp. Reg.', c=SP_REG_COLOR, ax=ax, s=MARKSIZE, marker=SS_MARKER)
ax = df[~ss].plot(kind='scatter', x="density", y='gflops/s|PSC', c=PSC_COLOR, ax=ax, s=MARKSIZE_DL, marker=DL_MARKER, label='DLMC')
ax = df[ss].plot(kind='scatter', x="density", y='gflops/s|PSC', c=PSC_COLOR, ax=ax, s=MARKSIZE, marker=SS_MARKER, label='SuiteSparse')
ax.set(xscale="log", yscale="linear")
ax.set_xlabel('Density (Log)')
ax.set_ylabel(None)
ax.set_title(f'500 DLMC (60%-95%) & 500 SuiteSparse', pad=10)
ax.axvline(x = 0.05, color = 'firebrick', linewidth=0.5)
ax.spines[['right', 'top']].set_visible(False)
ax.text(0.25, -0.27, "(b)", transform=ax.transAxes, size=18)

plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:g}'))
plt.gca().xaxis.set_minor_formatter(NullFormatter())
legend = ax.legend(prop={'size': 12})

for handle in legend.legend_handles:
    handle._sizes = [30]
    handle.set_color('black')

ax = axs[0, 2]
dff = df[df["cov|Sp. Reg."] >= 0.01]
def plot(ax, color, bias, data, label='nn'):
    geo_mean = _mean(data)
    ax.plot([x + bias - box_width * len(mcl)/2 for x in x_ticks], geo_mean, color=color, linewidth=1)
    return ax.boxplot(data, positions=[x + bias - box_width * len(mcl)/2 for x in x_ticks],
        notch=True, patch_artist=True,
        boxprops=dict(facecolor=color),
        capprops=dict(color=color),
        whiskerprops=dict(color=color),
        flierprops=dict(color=color, markeredgecolor=color, marker='o', markersize=0.5),
        medianprops=dict(color='black'),
        showfliers=True,
        #whis=(10,90),
        widths=0.15)

x_labels = ['0.01-0.1', '0.1-1', '1-10']
x_ranges = [(0.01, 0.1), (0.1, 1.0), (1, 10)]
x_ticks = [i+1 for i in range(len(x_labels))]

for i in range(len(x_labels)):
    data = df[(df['cov|Sp. Reg.']>=x_ranges[i][0])&(df['cov|Sp. Reg.']<x_ranges[i][1])]
    ss = data["ss"]
    print("Pct SS", x_labels[i], len(data[ss]) / len(data))

for mi, (method, color, label) in enumerate(mcl):
    data = [df[(df['cov|Sp. Reg.']>=x_ranges[i][0])&(df['cov|Sp. Reg.']<x_ranges[i][1])
                &(df[f'correct|{method}'] == "correct")
                &(~df[f'gflops/s|{method}'].isna())
                ][f'gflops/s|{method}'].tolist() for i in range(len(x_labels))]
    plots.append(plot(ax, color, box_width*mi, data))
    labels.append(label)

handles = [plot["boxes"][0] for plot in plots]

ax.set_xticks(x_ticks)
ax.set_xticklabels(x_labels, rotation=22)
ax.set_xlim([0.5, len(x_labels)+0.5])
ax.set_xlabel('CoV Working Set Size')
ax.set_title(f'500 DLMC (60%-95%) & 500 SuiteSparse', pad=10)
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)    
ax.set_ylim(limits)
ax.text(0.17, -0.27, "(c)", transform=ax.transAxes, size=18)


##
#   SAVE
##

plt.gcf().align_xlabels(axs[0, :])
plt.subplots_adjust(hspace=0.4, wspace=0.2) # For cascadelake
plt.margins(x=0)
plt.tight_layout(rect=(0,0,1,0.93)) # For cascadelake

savefig(f"/suitesparse_all.pdf")


##
#
##

baselines = ["MKL_Sparse", "MKL_Dense", "PSC"]

dff = all_dlmc_df
for baseline in baselines:
    for method, _, _ in mcl:
        spd = f'Speed-up {method} vs. {baseline}'
        dff[spd] = dff[f"time median|{baseline}"] / dff[f"time median|{method}"]
        #df.loc[df[spd] > 1e4, spd] = 1000
    print(method, "gmean speedup", baseline, gmean(dff[spd].tolist()))
    print(method, "pct faster (DLMC)", baseline, len(dff[dff[spd] > 1]) / len(dff))

baselines = ["MKL_Sparse", "PSC"]

print("=======")
dff = df[df.ss].copy()
for baseline in baselines:
    for method, _, _ in mcl:
        spd = f'Speed-up {method} vs. {baseline}'
        dff[spd] = dff[f"time median|{baseline}"] / dff[f"time median|{method}"]
        #df.loc[df[spd] > 1e4, spd] = 1000
        print(method, "pct faster (SS)", baseline, len(dff[dff[spd] > 1]) / len(dff))


baselines = ["PSC"]

for density in [1e-1, 1e-2, 1e-3, 1e-4]:
    dff = df[df.ss & (df["density"] <= density)].copy()
    for baseline in baselines:
        for method, _, _ in mcl:
            spd = f'Speed-up {method} vs. {baseline}'
            dff[spd] = dff[f"time median|{baseline}"] / dff[f"time median|{method}"]
            #df.loc[df[spd] > 1e4, spd] = 1000
        print(method, "pct faster (SS)    below:", density, baseline, len(dff[dff[spd] > 1]) / len(dff))
        print(method, "gmean speedup (ss) below:", density, baseline, gmean(dff[spd].tolist()))

