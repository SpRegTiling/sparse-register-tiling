import glob
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tools.paper.plotting.plot_utils import plot_save
from matplotlib import rc, rcParams
from tools.paper.plotting.plot_utils import *
from scipy.stats import gmean

from artifact.utils import *
from artifact.figure7_to_9.post_process_results import get_df, thread_list, bcols_list

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 15})
plt.rcParams["figure.figsize"] = (6, 4)
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0

def figure8():
    handles = []
    labels = []
    chipset = "cascadelake/"

    METHOD2 = {
        'cascadelake/': 'MKL_Dense',
        'raspberrypi/': 'ARMCL',
    }[chipset]

    METH2COLOR = {
        'cascadelake/': 'forestgreen',
        'raspberrypi/': 'lightcoral',
    }[chipset]


    METHOD2_name = {
        'cascadelake/': 'MKL (sgemm)',
        'raspberrypi/': 'ARMCL',
    }[chipset]

    METHOD3 = {
        'cascadelake/': 'ASpT',
        'raspberrypi/': '',
    }[chipset]

    METH3COLOR = {
        'cascadelake/': 'goldenrod',
        'raspberrypi/': '',
    }[chipset]

    METHOD3_name = {
        'cascadelake/': 'ASpT',
        'raspberrypi/': '',
    }[chipset]

    BASELINE = {
        'cascadelake/': 'MKL\n(spmm)',
        'raspberrypi/': 'XNNPACK\n(spmm, 16x1)',
    }[chipset]

    # x_labels = ['0.6', '0.7', '0.8', '0.9', '0.95', '0.98']

    numThreadsList = thread_list()
    numThreadsList.sort()
    bColsList = bcols_list()
    bColsList.sort()

    print(bColsList)

    df = get_df(bColsList[0], numThreadsList[0])

    x_labels = ['60%-69.9%', '70%-79.9%', '80%-89.9%', '90%-95%']
    x_ticks = [i+1 for i in range(len(x_labels))]

    def geometric_mean(list_in):
        geo_mean = []
        for sub_list in list_in:
            geo_mean.append(gmean(sub_list))
        
        return geo_mean

    def plot(ax, color, bias, data, label='nn'):
        geo_mean = geometric_mean(data)
        ax.plot(x_ticks, geo_mean, color=color)
        return ax.boxplot(data, positions=[x + bias for x in x_ticks],
            notch=True, patch_artist=True,
            boxprops=dict(facecolor=color),
            capprops=dict(color=color),
            whiskerprops=dict(color=color),
            flierprops=dict(color=color, markeredgecolor=color, marker='o', markersize=2),
            medianprops=dict(color='black'),
            widths=0.25)
        
    fig, axs = plt.subplots(len(numThreadsList), len(bColsList), figsize=(16,8), squeeze=False)

    methods = [
        "Sp. Reg.",
        "MKL_Dense",
        "ASpT"
    ]

    colors = ['steelblue', METH2COLOR, METH3COLOR]
    baseline = "MKL_Sparse"

    for i in range(len(numThreadsList)):
        for j in range(len(bColsList)):
            if (i, j) == (0, 0) or (i, j) == (0, 1) or (i, j) == (0, 2):
                limit = 2
            elif (i, j) == (0, 3) or (i, j) == (1, 2):
                limit = 3.5
            elif (i, j) == (1, 0):
                limit = 3
            elif (i, j) == (1, 1) or (i, j) == (1, 3):
                limit = 4
            if chipset == 'cascadelake/':
                limit = 5

            df = get_df(bColsList[j], numThreadsList[i])
            df = df[(df['pruningMethod'] == 'magnitude_pruning')|(df['pruningMethod'] == 'random_pruning')]


            # Note this will be updated to 
            #    `df["time cpu median|MKL_Sparse"] / min(df["time cpu median|ASpT"], df["time cpu median|ASpT_increased_parallelism"])`
            # in the final paper
            for method in methods:
                df[f'Speed-up {method} vs. {baseline}'] = df[f"time cpu median|{baseline}"] / df[f"time cpu median|{method}"]

            if (bColsList[j] == 256) and (numThreadsList[i] == 1):
                print(df[(df['sparsity']>=0*0.1+0.6)&(df['sparsity']<(0+1)*0.1+0.6)][f'Speed-up Sp. Reg. vs. {baseline}'])
                print(df[(df['sparsity']>=0*0.1+0.6)&(df['sparsity']<(0+1)*0.1+0.6)][f'correct|Sp. Reg.'])

            axs[i, j].plot([0.5, len(x_labels)+0.5],[1, 1], color='purple')
            plots = []

            for mi, method in enumerate(methods):
                data = [df[(df['sparsity']>=spBucket*0.1+0.6)&(df['sparsity']<(spBucket+1)*0.1+0.6)
                            &(df[f'correct|{method}'] == "correct")
                            &(~df[f'Speed-up {method} vs. {baseline}'].isna())
                            &(df[f'Speed-up {method} vs. {baseline}'] < limit)
                            ][f'Speed-up {method} vs. {baseline}'].tolist() for spBucket in range(len(x_labels))]
                if (bColsList[j] == 256) and (numThreadsList[i] == 1):
                    print(method, data[:5])
                plots.append(plot(axs[i, j], colors[mi], 0.15*mi, data))
            
            if (i, j) == (0, 0):
                handles.extend([plot["boxes"][0] for plot in plots])
                labels.extend(['Sparse Reg Tiling', METHOD2_name, 'ASpT'])
            axs[i, j].set_xticks(x_ticks)
            axs[i, j].set_xticklabels(x_labels, rotation=22)
            axs[i, j].set_xlim([0.5, len(x_labels)+0.5])
            if i == len(numThreadsList)-1:
                axs[i, j].set_xlabel('Sparsity')
            if j == 0:
                axs[i, j].set_ylabel(f'Speed-up vs {BASELINE}')
            axs[i, j].set_title(f'Threads={numThreadsList[i]}, B Columns={bColsList[j]}')
            axs[i, j].spines.right.set_visible(False)
            axs[i, j].spines.top.set_visible(False)
            axs[i, j].set_ylim([0, 4])
                
    fig.legend(handles, labels, loc='upper center', ncol=len(handles))
    plt.subplots_adjust(hspace=0.4, wspace=0.2) # For cascadelake
    #plt.subplots_adjust(hspace=0.2, wspace=0.2)
    plt.margins(x=0)
    plt.tight_layout(rect=(0,0,1,0.88)) # For cascadelake
    # plt.tight_layout(rect=(0,0,1,0.96))
    plt.savefig(PLOTS_DIR + "/figure8.jpg")

if __name__ == "__main__":
    figure8()