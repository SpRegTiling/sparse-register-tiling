import glob
import sys
import matplotlib.pyplot as plt
import numpy as np
from tools.paper.plotting.plot_utils import plot_save
from matplotlib import rc, rcParams
from tools.paper.plotting.plot_utils import *
from scipy.stats import gmean

SUBFOLDER = sys.argv[1] + '/'
NTHREADS = {
    'cascadelake/': 16,
    'raspberrypi/': 4,
}[SUBFOLDER]

METHOD2 = {
    'cascadelake/': r'MKL_Dense',
    'raspberrypi/': r'ARMCL',
}[SUBFOLDER]

df = load_dlmc_df(SUBFOLDER)

# print(df)

# df = filter(df, best_nano=True)
df = df[(df["sparsity"] <= 0.95) & (df["sparsity"] >= 0.7) & (df["n"] < 1024) & (df["m"] * df['k'] > 64 * 64)]
df = df[(df['matrixPath'].str.split('/').str[-3] == 'magnitude_pruning')|(df['matrixPath'].str.split('/').str[-3] == 'random_pruning')]

# print(df)

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
        flierprops=dict(color=color, markeredgecolor=color),
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
                    ]['Speed-up vs Sparse'].tolist() for spFolder in x_labels
        ]
        
        mklsp_data = [
            df[(df['sparsityFolder']==spFolder)
                     &(df["name"].str.contains(METHOD2, regex=True))
                     &(df['numThreads']==numThreadsList[i])
                     &(df['n']==bColsList[j])
                    ]['Speed-up vs Sparse'].tolist() for spFolder in x_labels
        ]
        
        # axs[i, j].gca().axhline(y=1.0, color='r', linestyle='-')
        axs[i, j].plot([0.5, len(x_labels)+0.5],[1, 1], color='purple')
        nano_p = plot(axs[i, j], 'steelblue', 0, nano_data, 'cuda')
        mklsp_p = plot(axs[i, j], 'forestgreen', 0.3, mklsp_data, 'blocked_ell')
        axs[i, j].set_xticks(x_ticks)
        axs[i, j].set_xticklabels(x_labels)
        axs[i, j].legend([nano_p["boxes"][0], mklsp_p["boxes"][0]], ['Nano', METHOD2], loc='upper right')
        axs[i, j].set_xlim([0.5, len(x_labels)+0.5])
        if i == len(numThreadsList)-1:
            axs[i, j].set_xlabel('Sparsity')
        if j == 0:
            axs[i, j].set_ylabel('Speed-up vs Sparse')
        axs[i, j].set_title(f'Threads={numThreadsList[i]}, B Columns={bColsList[j]}')
        
plt.subplots_adjust(hspace=0.3, wspace=0.2)

plot_save(f"boxes/{SUBFOLDER}/vs_sparse_jitter")


# ax = df.plot.scatter(x='gflops', y='Speed-up vs Dense', c='sparsity', colormap='cividis', alpha=0.5, s=1)
# ax.set_xscale('log')
# ax.axhline(y=1.0, color='r', linestyle='-')

# plt.ylabel('Speed-up vs MKL Dense')
# plt.xlabel('Problem Size (GFLOPs)')
# plot_save(f"scatters/{SUBFOLDER}/vs_dense")
