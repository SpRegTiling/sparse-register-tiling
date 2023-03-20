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

def figure8():
    handles = []
    labels = []

    chipset = "cascade"
    
    if chipset == "raspberrypi":
        numThreads = 4
    else:
        numThreads = 1
    bcols = 128
    
    df = read_cache(chipset, "all", bcols=bcols, threads=numThreads)

    pruning_methods = []
    models = df["model"].unique()
    for model in models:
        dff = filter(df, model=model)
        pmethods = sorted(dff["pruningMethod"].unique())
        # move l0 and extended to end
        pmethods.append(pmethods[0])
        pruning_methods.append(pmethods[1:])
    max_prune_count = max([len(x) for x in pruning_methods])

    def geometric_mean(list_in):
        geo_mean = []
        for sub_list in list_in:
            geo_mean.append(mean(sub_list))
        
        return geo_mean

    if chipset == "raspberrypi":
        mcl = arm_mcl
        limits = [(0, 2.5), (0, 8)]
    else:
        mcl = intel_mcl_no_aspt
        limits = [(0, 80), (0, 800)]

    BOX_WIDTH = 0.15

    def plot(ax, color, bias, box_width, data, label='nn'):
        geo_mean = geometric_mean(data)
        ax.plot([x + bias - box_width * len(mcl)/2 for x in x_ticks], geo_mean, color=color, linewidth=1)
        return ax.boxplot(data, positions=[x + bias - box_width * len(mcl)/2 for x in x_ticks],
            notch=True, patch_artist=True,
            boxprops=dict(facecolor=color),
            capprops=dict(color=color),
            whiskerprops=dict(color=color),
            flierprops=dict(color=color, markeredgecolor=color, marker='o', markersize=0.5),
            medianprops=dict(color='black'),
            showfliers=True,
            widths=box_width)
        
    fig, axs = plt.subplots(len(models), max_prune_count, figsize=(16,8), squeeze=False)

    for i in range(len(models)):
        for j in range(max_prune_count):
            df = read_cache(chipset, "all", bcols=bcols, threads=numThreads)
            df = filter(df, pruningMethod=pruning_methods[i][j], model=models[i])
            pruning_method = pruning_methods[i][j]
            if pruning_method in ["magnitude_pruning", "random_pruning"]:
                plot_type = "box"
                sparsities = sorted(df["pruningModelTargetSparsity"].unique())
                x_labels = [f'{round(x*100)}%'for x in sparsities]
                x_ticks = [i+1 for i in range(len(x_labels))]
                sparsity_buckets =  [(x-0.01, x+0.01) for x in sparsities]
            elif pruning_method in ["l0_regularization", "variational_dropout"]:
                plot_type = "box"
                x_labels = ['60%-69.9%', '70%-79.9%', '80%-89.9%', '90%-95%']
                sparsity_buckets =  [(x*0.1 + 0.6, (x+1)*0.1 + 0.6) for x in range(len(x_labels))]
                x_ticks = [i+1 for i in range(len(x_labels))]
            else:
                plot_type = "box"
                x_labels = ['80%', '91%']
                sparsity_buckets =  [(0.79, 0.81), (0.9, 0.92)]
                x_ticks = [i+1 for i in range(len(x_labels))]
                      
            if chipset == "cascade": df = compute_aspt_best(df)

            # for method, _, _ in mcl:
            #     df[f'Speed-up {method} vs. {baseline}'] = df[f"time cpu median|{baseline}"] / df[f"time cpu median|{method}"]

            # axs[i, j].plot([0.5, len(x_labels)+0.5],[1, 1], color='purple')
            plots = []
            labels = []
            
            for mi, (method, color, label) in enumerate(mcl):
                if plot_type == "box":
                    box_width = BOX_WIDTH * len(sparsity_buckets) / 4

                    data = [df[(df['sparsity']>=spBucket[0])&(df['sparsity']<spBucket[1])
                                &(df[f'correct|{method}'] == "correct")
                                &(~df[f'gflops/s|{method}'].isna())
                                ][f'gflops/s|{method}'].tolist() for spBucket in sparsity_buckets]
                    plots.append(plot(axs[i, j], color, box_width*mi, box_width, data))
                    labels.append(label)
                else:
                    dff = df[(df[f'correct|{method}'] == "correct")
                                &(~df[f'gflops/s|{method}'].isna())]
                    dff.plot.scatter(x='sparsity', y=f'gflops/s|{method}', c=color, ax=axs[i, j])
                    
            MODEL_STRINGS = {
                "rn50": "ResNet50",
                "transformer": "Transformer"
            }
            
            PRUNING_STRINGS = {
                "magnitude_pruning": "Magnitude Pruning",
                "random_pruning": "Random Pruning",
                "variational_dropout": "Variational Dropout",
                "extended_magnitude_pruning": "Extd. Magnitude Pruning",
                "l0_regularization": "l0 Regularization"
            }
            
            
            assert len(plots) > 0            
            handles = [plot["boxes"][0] for plot in plots if "boxes" in plot]
            
            ax = axs[i, j]
            ax.set_xticks(x_ticks)
            axs[i, j].set_xticklabels(x_labels, rotation=22)
            axs[i, j].set_xlim([0.5, len(x_labels)+0.5])
            if i == len(models) - 1:
                axs[i, j].set_xlabel('Sparsity')
            if j == 0:
                axs[i, j].set_ylabel(f'Required GFLOP/s')
                axs[i, j].text(-0.2, 1.2, f'{MODEL_STRINGS[models[i]]}',
                    horizontalalignment='left',
                    verticalalignment='bottom',
                    transform=ax.transAxes,
                    fontsize=20)    
            
            axs[i, j].set_title(f'{PRUNING_STRINGS[pruning_methods[i][j]]}', fontsize=16)
            axs[i, j].spines.right.set_visible(False)
            axs[i, j].spines.top.set_visible(False)    
            axs[i, j].set_ylim(limits[0])
    
    for i in range(len(models)):
        plt.gcf().align_xlabels(axs[i, :])
    
    fig.legend(handles, labels, loc='upper center', ncol=len(handles))
    plt.subplots_adjust(hspace=0.05, wspace=0.05) # For cascadelake
    plt.margins(x=0)
    plt.tight_layout(rect=(0,0,1,1)) # For cascadelake
    savefig(f"/figure_prune_v2_{chipset}.pdf")

if __name__ == "__main__":
    figure8()