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
from functools import reduce
from scipy.io import mmwrite
from functools import partial

import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

from spmm_benchmarks.loaders.dlmc import DLMCLoader
from spmm_benchmarks.loaders.suitesparse import SuiteSparseLoader
from spmm_benchmarks.loaders.filelist import FilelistLoader
from spmm_benchmarks.loaders.load import load_dense, load_csr, load_coo

torch.set_grad_enabled(False)
os.environ['DLMC_ROOT'] = '/sdb/datasets/dlmc'


def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom  # or / in Python 2


ss_loader = SuiteSparseLoader(loader=load_dense)
ss_mini_loader = FilelistLoader(SCRIPT_DIR + "/../../tools/filelists/ss_mini.txt", loader=load_dense)

dlmc_loader = DLMCLoader(
    file_list=SCRIPT_DIR + "/../../tools/filelists/hybrid_file_list_70.txt",
    loader=load_dense
)


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


def plot_histograms(vec_height, sparsity, counts, matrix_name):
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

    plt.bar(range(vec_height+1), scaled_probs_sums)
    plt.step(np.arange(vec_height+1) + 0.5, scaled_random_probs_sums, color='red')
    plt.xticks(range(vec_height+1), rotation=90)
    plt.title(matrix_name)
    plt.ylabel('Probability of Nonzero Existing in Pattern')
    plt.xlabel('Size of Pattern in Nonzeros')
    plt.gcf().set_size_inches(20, 10)
    plt.savefig(PLOT_FOLDER + matrix_name + f'_sums_SOP{vec_height}_{sparsity}.png')
    plt.clf()

    plt.bar(range(2**vec_height), scaled_probs)
    plt.step(np.arange(0, 2**vec_height) + 0.5, scaled_random_probs, color='red')
    plt.xticks(np.arange(0, 2**vec_height, 3), [f'0b{x:08b}'for x in sorted_patterns[::3]], rotation=90)
    plt.title(matrix_name)
    plt.ylabel('Probability of Nonzero Existing in Pattern')
    plt.xlabel('Pattern')
    plt.gcf().set_size_inches(20, 10)
    plt.savefig(PLOT_FOLDER + matrix_name + f'_SOP{vec_height}_{sparsity}.png')
    plt.clf()


PLOT_FOLDER = SCRIPT_DIR + "/../../plots/histograms/"

global_pat_counts = {}

def compute_stats(vec_height, matrix_name, matrix: torch.Tensor):
    global global_pat_counts

    counts = compute_histogram(vec_height, matrix)

    sparsity = round((1 - matrix.sum() / np.prod(matrix.shape)).item(), 2)
    key = (vec_height, sparsity)

    if key not in global_pat_counts:
        global_pat_counts[key] = counts
    else:
        global_pat_counts[key] += counts

    plot_histograms(vec_height, sparsity, counts, matrix_name)
    print((vec_height, sparsity), global_pat_counts[key])


# for matrix, path in ss_loader:
#     compute_stats(matrix)

# for matrix, path in ss_mini_loader:
#     matrix_name = os.path.splitext(os.path.basename(path))[0]
#     print(matrix_name)
#     compute_stats(4, matrix_name, matrix)
#     compute_stats(8, matrix_name, matrix)

# plot_histograms(4, 0.7, global_pat_counts[(4, 0.7)], 'global')
# plot_histograms(8, 0.7, global_pat_counts[(8, 0.7)], 'global')


# def plot_random_histograms(vec_height):
#     patterns = np.arange(0, 2**vec_height)
#     sorted_patterns = sorted(patterns, key=lambda x: gmpy.popcount(int(x)) * (2**vec_height) + x)
#
#     popcount_counts = [ncr(vec_height, x) for x in range(vec_height + 1)]
#     popcount_offsets = np.concatenate((np.zeros(1, dtype=int), np.cumsum(popcount_counts).astype(int)))
#
#     width = 0.2
#
#     x = np.arange(vec_height+1, dtype=int)
#
#     for sparsity in [0.7, 0.8, 0.9, 0.95]:
#         rand_probs = np.vectorize(partial(pattern_random_prob, vec_height=vec_height, sparsity=sparsity))(sorted_patterns)
#         pat_nnz = np.vectorize(pattern_nnz)(sorted_patterns)
#
#         scaled_random_probs = (rand_probs * pat_nnz) / np.sum(rand_probs * pat_nnz)
#         scaled_random_probs_sums = sum_offsets(scaled_random_probs, popcount_offsets)
#
#         plt.plot(x, scaled_random_probs_sums, label=str(sparsity))
#         #x = x + width
#
#     plt.title('Random')
#     plt.ylabel('Probability of Nonzero Existing in Pattern')
#     plt.xlabel('Size of Pattern in Nonzeros')
#     plt.legend()
#     plt.gcf().set_size_inches(20, 10)
#     plt.savefig(PLOT_FOLDER + f'random_sums_{vec_height}.png')
#     plt.clf()
#
#
# plot_random_histograms(4)
# plot_random_histograms(8)
# print(global_pat_counts)


def gen_csv(vec_height):
    patterns = np.arange(0, 2**vec_height)
    sorted_patterns = sorted(patterns, key=lambda x: gmpy.popcount(int(x)) * (2**vec_height) + x)

    popcount_counts = [ncr(vec_height, x) for x in range(vec_height + 1)]
    popcount_offsets = np.concatenate((np.zeros(1, dtype=int), np.cumsum(popcount_counts).astype(int)))

    width = 0.2

    x = np.arange(vec_height+1, dtype=int)

    for sparsity in [0.7, 0.8, 0.9, 0.95]:
        rand_probs = np.vectorize(
            partial(pattern_random_prob, vec_height=vec_height, sparsity=sparsity))(sorted_patterns)
        pat_nnz = np.vectorize(pattern_nnz)(sorted_patterns)

        scaled_random_probs = (rand_probs * pat_nnz) / np.sum(rand_probs * pat_nnz)
        scaled_random_probs_sums = sum_offsets(scaled_random_probs, popcount_offsets)

        plt.plot(x, scaled_random_probs_sums, label=str(sparsity))

        with open(PLOT_FOLDER + f'random_probs_SOP{vec_height}_{round(sparsity*100)}%_sparsity.txt', 'w+') as f:
            for pattern, prob in zip(sorted_patterns, rand_probs):
                f.write(f'0b{pattern:08b}, {prob}\n')


gen_csv(4)
gen_csv(8)
print(global_pat_counts)