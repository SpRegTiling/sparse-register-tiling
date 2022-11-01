from sbench.loaders.load import load_dense, load_csr, load_coo
from functools import partial
from typing import Dict, Tuple, List, Union

import numpy as np
import gmpy
import torch


def pattern_code(vec):
    vec = vec.to(dtype=torch.bool).to(dtype=torch.int)
    pat = 0
    for idx, i in enumerate(vec):
        pat |= i.item() << idx
    return pat


def compute_histogram(M_r, matrix):
    pattern_count = [0] * 2 ** M_r

    for i in range(0, matrix.shape[0], M_r):
        for row in matrix[i:i + M_r].t():
            pattern = pattern_code(row)
            pattern_count[pattern] += 1

    return np.array(pattern_count) / (matrix.shape[1] * matrix.shape[0] // M_r)


def pattern_random_prob(pattern, vec_height, sparsity):
    nnz = gmpy.popcount(int(pattern))
    return (sparsity ** (vec_height - nnz)) * ((1 - sparsity) ** nnz)


def pattern_nnz(pattern):
    nnz = gmpy.popcount(int(pattern))
    return nnz


def sum_offsets(vector, offsets):
    return np.array([np.sum(vector[offsets[i-1]:offsets[i]]) for i in range(1, len(offsets))])


def extract_distribution(M_r, filepaths: Union[str, List[str]]):
    if type(filepaths) == str:
        filepaths = [filepaths]

    current_counts = np.zeros(2 ** M_r)
    for filepath in filepaths:
        current_counts += compute_histogram(M_r, load_dense(filepath))

    return current_counts / len(filepaths)


def save_distribution(filepath: str, distribution: callable):
    return None


def read_distribution(filename):
    pass
