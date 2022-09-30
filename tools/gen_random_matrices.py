import os
import torch
import zlib
import numpy as np
import scipy
from scipy.io import mmwrite
import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

DATASET_DIR = os.environ.get("DATASET_DIR", "/sdb/datasets")
file_generated = []


MATRIX_DIMS = [768, 768]

for sp in np.linspace(0.01, 1.0, 100):
    mat = scipy.sparse.random(MATRIX_DIMS[0], MATRIX_DIMS[1], density=1-sp, dtype=np.float32)
    mmwrite(f'{DATASET_DIR}/random/random_{int(round(sp, 2) *100)}_{MATRIX_DIMS[0]}_{MATRIX_DIMS[1]}.mtx', mat)
    file_generated.append(f'random/random_{int(round(sp, 2) *100)}_{MATRIX_DIMS[0]}_{MATRIX_DIMS[1]}.mtx')

with open(f'{SCRIPT_DIR}/filelists/random_sweep.txt', 'w') as f:
    f.write('\n'.join(file_generated))
    f.write('\n')
