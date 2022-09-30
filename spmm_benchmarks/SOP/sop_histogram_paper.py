import ssgetpy
import numpy as np
import torch
import copy
import pandas as pd
import xformers
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import math
import gmpy
import operator as op
import random
from functools import reduce
from scipy.io import mmwrite
from functools import partial
import matplotlib.ticker as mtick
import pickle
from matplotlib.lines import Line2D


import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

from spmm_benchmarks.loaders.dlmc import DLMCLoader
from spmm_benchmarks.loaders.suitesparse import SuiteSparseLoader
from spmm_benchmarks.loaders.filelist import FilelistLoader
from spmm_benchmarks.loaders.load import load_dense, load_csr, load_coo


random.seed(10)
MATRICES_TO_PLOT = 15

matrix_list = ssgetpy.search(rowbounds=[512, 4*4096], colbounds=[512, 4*4096],limit=MATRICES_TO_PLOT*50)
print(len(matrix_list))
matrix_ids = [matrix.id for matrix in matrix_list]
random.shuffle(matrix_ids)
filelist = SCRIPT_DIR + "/../../tools/filelists/hybrid_file_list_80.txt"

torch.set_grad_enabled(False)
os.environ['DLMC_ROOT'] = '/sdb/datasets/dlmc'

ss_loader = SuiteSparseLoader(matrix_ids=matrix_ids[:MATRICES_TO_PLOT], loader=load_dense)
ml_loader = DLMCLoader(file_list=filelist, loader=load_dense, random=MATRICES_TO_PLOT)


def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom  # or / in Python 2


def pattern_code(vec):
    vec = vec.to(dtype=torch.bool).to(dtype=torch.int)
    pat = 0
    for idx, i in enumerate(vec):
        pat |= i.item() << idx
    return pat


def compute_histogram(vec_height, matrix):
    pattern_count = [0] * 2**vec_height

    for i in range(0, matrix.shape[0], vec_height):
        for row in matrix[i:i+vec_height].t():
            pattern = pattern_code(row)
            pattern_count[pattern] += 1

    return np.array(pattern_count)


def pattern_random_prob(pattern, vec_height, sparsity):
    nnz = gmpy.popcount(int(pattern))
    return (sparsity ** (vec_height - nnz)) * ((1 - sparsity) ** nnz)


def pattern_nnz(pattern):
    nnz = gmpy.popcount(int(pattern))
    return nnz


def sum_offsets(vector, offsets):
    return np.array([np.sum(vector[offsets[i-1]:offsets[i]]) for i in range(1, len(offsets))])


def dist_from_random(vec_height, sparsity, counts, matrix_name):
    total_count = np.sum(counts)
    normalized_counts = counts / total_count
    patterns = np.arange(0, 2**vec_height)

    sorted_patterns = sorted(patterns, key=lambda x: gmpy.popcount(int(x)) * (2**vec_height) + x)
    sorted_normalized_counts = np.take(normalized_counts, sorted_patterns)

    popcount_counts = [ncr(vec_height, x) for x in range(vec_height + 1)]
    popcount_offsets = np.concatenate((np.zeros(1, dtype=int), np.cumsum(popcount_counts).astype(int)))

    rand_probs = np.vectorize(partial(pattern_random_prob, vec_height=vec_height, sparsity=sparsity))(sorted_patterns)
    pat_nnz = np.vectorize(pattern_nnz)(sorted_patterns)

    scaled_probs = (sorted_normalized_counts * pat_nnz) / np.sum(sorted_normalized_counts * pat_nnz)
    scaled_random_probs = (rand_probs * pat_nnz) / np.sum(rand_probs * pat_nnz)

    scaled_probs_sums = sum_offsets(scaled_probs, popcount_offsets)
    scaled_random_probs_sums = sum_offsets(scaled_random_probs, popcount_offsets)

    return scaled_probs - scaled_random_probs


PLOT_FOLDER = SCRIPT_DIR + "/../../plots/histograms/"
global_pat_counts = {}


def compute_stats(vec_height, matrix_name, matrix: torch.Tensor):
    global global_pat_counts
    matrix = matrix.to(dtype=torch.bool).to(dtype=torch.int)
    counts = compute_histogram(vec_height, matrix)

    sparsity = (1 - matrix.sum() / np.prod(matrix.shape)).item()
    key = (vec_height, sparsity)

    if key not in global_pat_counts:
        global_pat_counts[key] = counts
    else:
        global_pat_counts[key] += counts

    return dist_from_random(vec_height, sparsity, counts, matrix_name)
    #print((vec_height, sparsity), global_pat_counts[key])


    # plt.bar(range(vec_height+1), scaled_probs_sums)
    # plt.step(np.arange(vec_height+1) + 0.5, scaled_random_probs_sums, color='red')
    # plt.xticks(range(vec_height+1), rotation=90)
    # plt.title(matrix_name)
    # plt.ylabel('Probability of Nonzero Existing in Pattern')
    # plt.xlabel('Size of Pattern in Nonzeros')
    # plt.gcf().set_size_inches(20, 10)
    # plt.savefig(PLOT_FOLDER + matrix_name + f'_sums_SOP{vec_height}_{sparsity}.png')
    # plt.clf()
    #
    # plt.bar(range(2**vec_height), scaled_probs)
    # plt.step(np.arange(0, 2**vec_height) + 0.5, scaled_random_probs, color='red')
    # plt.xticks(np.arange(0, 2**vec_height, 3), [f'0b{x:08b}'for x in sorted_patterns[::3]], rotation=90)
    # plt.title(matrix_name)
    # plt.ylabel('Probability of Nonzero Existing in Pattern')
    # plt.xlabel('Pattern')
    # plt.gcf().set_size_inches(20, 10)
    # plt.savefig(PLOT_FOLDER + matrix_name + f'_SOP{vec_height}_{sparsity}.png')
    # plt.clf()

vec_height = 8


if not os.path.exists(SCRIPT_DIR + "/../../data/hist_data.pkl"):
    ss = []
    for matrix, path in ss_loader:
        ss.append(compute_stats(vec_height, path.split('/')[-1], matrix))

    ml = []
    for matrix, path in ml_loader:
        ml.append(compute_stats(vec_height, path.split('/')[-1], matrix))

    pickle.dump((ss, ml), open(SCRIPT_DIR + "/../../data/hist_data.pkl", "wb"))

else:
    ss, ml = pickle.load(open(SCRIPT_DIR + "/../../data/hist_data.pkl", "rb"))

patterns = np.arange(0, 2**vec_height)
sorted_patterns = sorted(patterns, key=lambda x: gmpy.popcount(int(x)) * (2**vec_height) + x)
rand_probs = np.vectorize(partial(pattern_random_prob, vec_height=vec_height, sparsity=0.8))(sorted_patterns)

popcount_counts = [ncr(vec_height, x) for x in range(vec_height + 1)]
popcount_offsets = np.concatenate((np.zeros(1, dtype=int), np.cumsum(popcount_counts).astype(int)))
pat_nnz = np.vectorize(pattern_nnz)(sorted_patterns)
scaled_random_probs = (rand_probs * pat_nnz) / np.sum(rand_probs * pat_nnz)

fig, axes = plt.subplots(2, 1, sharex='col')
fig.suptitle(f'Percentage of Nonzero Covered by Specific Nano-kernel Patterns ($M_r$ = {vec_height})')
fig.set_size_inches(16, 6)

axes[0].step(np.arange(0, 2**vec_height) + 0.5, scaled_random_probs * 100, color='black')
axes[0].set(ylabel='Random Uniform Matrix\n(80% Sparsity)')
#axes[0].set(xlabel='Nano-kernel Pattern')
axes[0].yaxis.set_major_formatter(mtick.PercentFormatter())


for _ss in ss:
    axes[1].step(np.arange(0, 2**vec_height) + 0.5, _ss * 100, color='red', label='SuiteSparse')

for _ml in ml:
    axes[1].step(np.arange(0, 2**vec_height) + 0.5, _ml * 100, color='blue', label='ML (Google Dataset)')

custom_lines = [Line2D([0], [0], color='red', lw=2),
                Line2D([0], [0], color='blue', lw=2)]

axes[1].legend(custom_lines, ['SuiteSparse', 'ML (Google Dataset, 80% Sparsity)'])
axes[1].set(ylabel='Difference Compared to\nUniform Random Matrix')
axes[1].set(xlabel='Nano-kernel Pattern')
axes[1].yaxis.set_major_formatter(mtick.PercentFormatter())
axes[1].set_xticks(np.arange(0, 2**vec_height, 3))
axes[1].set_xticklabels([f'{x:08b}'.replace('0', ' ').replace('1', '*') for x in sorted_patterns[::3]], rotation=90)

plt.tight_layout()
fig.show()
fig.savefig(PLOT_FOLDER + f'paper_hist.png')
plt.show(block=True)

# for matrix, path in ss_mini_loader:
#     matrix_name = os.path.splitext(os.path.basename(path))[0]
#     print(matrix_name)
#     compute_stats(4, matrix_name, matrix)
#     compute_stats(8, matrix_name, matrix)

# plot_histograms(4, 0.7, global_pat_counts[(4, 0.7)], 'global')
# plot_histograms(8, 0.7, global_pat_counts[(8, 0.7)], 'global')