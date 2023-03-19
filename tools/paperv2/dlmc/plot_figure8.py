import glob
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tools.paper.plotting.plot_utils import plot_save
from matplotlib import rc, rcParams
from tools.paper.plotting.plot_utils import *
from scipy.stats import gmean
from numpy import mean

from tools.paperv2.dlmc.utils import *

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 15})
plt.rcParams["figure.figsize"] = (6, 4)
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0

def figure8(chipset = "cascade"):
    handles = []
    labels = []
    # x_labels = ['0.6', '0.7', '0.8', '0.9', '0.95', '0.98']
    
    if chipset == "raspberrypi":
        numThreadsList = [1, 4]
    else:
        numThreadsList = [1, 20]
    numThreadsList.sort()
    bColsList = [32, 128, 256, 512]
    bColsList.sort()
    
    df = read_cache(chipset, "all", bcols=bColsList[0], threads=numThreadsList[0])

    x_labels = ['60%-69.9%', '70%-79.9%', '80%-89.9%', '90%-95%']
    x_ticks = [i+1 for i in range(len(x_labels))]

    def _mean(list_in):
        m = []
        for sub_list in list_in:
            m.append(mean(sub_list))
        
        return m

    if chipset == "raspberrypi":
        mcl = arm_mcl
        limits = [(0, 2.5), (0, 8)]
    else:
        mcl = intel_mcl
        limits = [(0, 80), (0, 800)]

    box_width = 0.15

    def plot(ax, color, bias, data, label='nn'):
        m = _mean(data)
        ax.plot([x + bias - box_width * len(mcl)/2 for x in x_ticks], m, color=color, linewidth=1)
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
        
    fig, axs = plt.subplots(len(numThreadsList), len(bColsList), figsize=(16,7.5), squeeze=False)

    for i in range(len(numThreadsList)):
        for j in range(len(bColsList)):
            df = read_cache(chipset, "all", bcols=bColsList[j], threads=numThreadsList[i])
            if chipset == "cascade": df = compute_aspt_best(df)

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
                plots.append(plot(axs[i, j], color, box_width*mi, data))
                labels.append(label)
            
            handles = [plot["boxes"][0] for plot in plots]

            axs[i, j].set_xticks(x_ticks)
            axs[i, j].set_xticklabels(x_labels, rotation=18)
            axs[i, j].set_xlim([0.5, len(x_labels)+0.5])
            if i == len(numThreadsList)-1:
                axs[i, j].set_xlabel('Sparsity')
            if j == 0:
                axs[i, j].set_ylabel(f'Required GFLOP/s')
            axs[i, j].set_title(f'Threads={numThreadsList[i]}, B Columns={bColsList[j]}')
            axs[i, j].spines.right.set_visible(False)
            axs[i, j].spines.top.set_visible(False)    
            axs[i, j].set_ylim(limits[0] if numThreadsList[i] == 1 else limits[1])
            
            if chipset == "raspberrypi" and bColsList[j] == 32:
                axs[i, j].set_ylim((0,6) if numThreadsList[i] == 1 else (0,20))
            
                
    fig.legend(handles, labels, loc='upper center', ncol=len(handles))
    plt.subplots_adjust(hspace=0.4, wspace=0.1) # For cascadelake
    plt.gcf().align_ylabels(axs[:, 0])
    plt.margins(x=0)
    plt.tight_layout(rect=(0,0,1,0.95)) # For cascadelake
    savefig(f"/figure8_v2_{chipset}.pdf")

if __name__ == "__main__":
    figure8("cascade")
    figure8("raspberrypi")