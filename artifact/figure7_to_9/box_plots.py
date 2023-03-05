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


def figure8():
    def post_process(df):
        nano_methods = df['name'].str.contains(r'M[0-9]N[0-9]', regex=True)
        df["orig_name"] = df['name']
        df.loc[nano_methods, 'name'] = "Sp. Reg."
        df["is_nano"] = df['name'].str.contains("Sp. Reg.")
        df["is_aspt"] = df['name'].str.contains("ASpT")
        df["sparsity"] = round(1 - df["nnz"] / (df["m"] * df["k"]), 2)
        df["sparsity_raw"] = 1 - df["nnz"] / (df["m"] * df["k"])
        df["pruningMethod"] = df["matrixPath"].str.split("/").str[-3]
        df["model"] = df["matrixPath"].str.split("/").str[-4]
        df["sparsityFolder"] = df["matrixPath"].str.split("/").str[-2]
        df["matrixName"] = df["matrixPath"].str.split("/").str[-1].str.replace(".mtx", "").str.replace(".smtx", "")
        df["matrixId"] = df["sparsityFolder"] + "|" + df["model"] + "|" + df["pruningMethod"] + "|" + df["matrixName"]
        df["sparsityFolder"] = round(df["sparsityFolder"].astype(float), 2)

        return df

    def pivot(df, bcols, num_threads):
        df = filter(df, n=bcols, numThreads=num_threads)
        df = df.reset_index(drop=True)

        sparsity_buckets = pd.IntervalIndex.from_tuples([(0.0, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)])
        df["sparsity_buckets"] = pd.cut(df["sparsity"], sparsity_buckets)
        df["flops"] = 2 * df["n"] * df["nnz"]
        df["gflops/s"] = (df["flops"] / (df["time median"]/1e6)) / 1e9
        df["Method"] = df["name"]

        dfw = pd.pivot(df, index=["matrixId", "m", "k", "nnz", "n", "sparsity"], columns=["Method"],
                    values=["gflops/s", "time cpu median", "correct"])

        dfw.index.names = ['Matrix', "Rows", "Cols", "NNZ", "Bcols", "Sparsity"]
        dfw.columns = dfw.columns.get_level_values(level=1)
        return dfw

    def box_plots(df, chipset):
        METHOD2 = {
            'cascadelake': 'MKL_Dense',
            'raspberrypi': 'ARMCL',
        }[chipset]

        METHOD2_name = {
            'cascadelake': 'MKL (sgemm)',
            'raspberrypi': 'ARMCL',
        }[chipset]

        BASELINE = {
            'cascadelake': 'MKL (spmm)',
            'raspberrypi': 'XNNPACK (spmm, 16x1)',
        }[chipset]

        # df = filter(df, best_nano=True)
        df = df[(df["sparsity"] <= 0.95) & (df["sparsity"] >= 0.6) & (df["n"] < 1024) & (df["m"] * df['k'] > 64 * 64)]
        df = df[(df['matrixPath'].str.split('/').str[-3] == 'magnitude_pruning')|(df['matrixPath'].str.split('/').str[-3] == 'random_pruning')]

        # x_labels = ['0.6', '0.7', '0.8', '0.9', '0.95', '0.98']
        x_labels = list(df['sparsityFolder'].unique())
        x_labels.sort()
        numThreadsList = list(df['numThreads'].unique())
        numThreadsList.sort()
        bColsList = list(df['n'].unique())
        bColsList.sort()
        x_ticks = [i+1 for i in range(len(x_labels))]

        print(numThreadsList)
        print(bColsList)

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
            
        fig, axs = plt.subplots(len(numThreadsList), len(bColsList), figsize=(16,8))
        # fig, axs = plt.subplots()

        for i in range(len(numThreadsList)):
            for j in range(len(bColsList)):
                print(f'i, j: {i}, {j}')
                nano_data = [
                    df[(df['sparsityFolder']==spFolder)
                            &(df["is_nano"] == True)&(df['best_nano'] == True)
                            &(df['numThreads']==numThreadsList[i])
                            &(df['n']==bColsList[j])
                            &(~df['Speed-up vs Sparse'].isna())
                            &(df['Speed-up vs Sparse'] < 3)
                            ]['Speed-up vs Sparse'].tolist() for spFolder in x_labels
                ]
                
                sp_data = [
                    df[(df['sparsityFolder']==spFolder)
                            &(df["name"].str.contains(METHOD2, regex=True))
                            &(df['numThreads']==numThreadsList[i])
                            &(df['n']==bColsList[j])
                            &(df['Speed-up vs Sparse'] < 3)
                            ]['Speed-up vs Sparse'].tolist() for spFolder in x_labels
                ]
                
                # axs[i, j].gca().axhline(y=1.0, color='r', linestyle='-')
                axs[i, j].plot([0.5, len(x_labels)+0.5],[1, 1], color='purple')
                nano_p = plot(axs[i, j], 'steelblue', 0, nano_data, 'cuda')
                sp_p = plot(axs[i, j], 'lightcoral', 0.3, sp_data, 'blocked_ell')
                axs[i, j].set_xticks(x_ticks)
                axs[i, j].set_xticklabels(x_labels)
                axs[i, j].legend([nano_p["boxes"][0], sp_p["boxes"][0]], ['Sparse Register Tiling', METHOD2_name], loc='upper right')
                axs[i, j].set_xlim([0.5, len(x_labels)+0.5])
                if i == len(numThreadsList)-1:
                    axs[i, j].set_xlabel('Sparsity')
                if j == 0:
                    axs[i, j].set_ylabel(f'Speed-up vs {BASELINE}')
                axs[i, j].set_title(f'Threads={numThreadsList[i]}, B Columns={bColsList[j]}')
                
        plt.subplots_adjust(hspace=0.3, wspace=0.2)

    df = pd.read_csv(RESULTS_DIR + 'figure7_to_9_results.txt')
    df = post_process(df)
    print(df)
    df = pivot(df, 128, 1)
    print(df)

if __name__ == "__main__":
    figure8()