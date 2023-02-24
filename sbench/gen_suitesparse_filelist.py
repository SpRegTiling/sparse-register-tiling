import numpy as np
import torch
import random
from typing import List
import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

from scipy.stats import variation
from sbench.loaders.suitesparse import SuiteSparseLoader
from sbench.loaders.load import load_dense, load_csr, load_coo

import ssgetpy


NUM_LARGE_MATRICES = 50
NUM_SMALL_MATRICES = 50

random.seed(42)

# print("=============== SMALL  MATRICES ===================")
#
# matrix_list = ssgetpy.search(rowbounds=[int(0), int(1e5)],
#                              colbounds=[int(0), int(1e5)],
#                              nzbounds=[int(0), int(1e10)],
#                              limit=NUM_SMALL_MATRICES*5)
# matrix_ids = [matrix.id for matrix in matrix_list]
# random.shuffle(matrix_ids)
#
# ss_loader = SuiteSparseLoader(matrix_ids=matrix_ids[:NUM_SMALL_MATRICES], loader=load_csr)
#
# with open(SCRIPT_DIR + "/../tools/filelists/ss_small.txt", "w+") as f:
#     for _, path in ss_loader:
#         partial_path = "ss" + "ss".join(path.split('ss')[1:])
#         f.write(partial_path + "\n")


print("=============== LARGE MATRICES ALL ===================")

matrix_list = ssgetpy.search(rowbounds=[int(1e5), int(1e16)],
                             colbounds=[int(1e5), int(1e16)],
                             nzbounds=[int(1e6), int(1e9)],
                             limit=10000)
matrix_ids = [matrix.id for matrix in matrix_list]
random.shuffle(matrix_ids)

ss_loader = SuiteSparseLoader(matrix_ids=matrix_ids, loader=load_csr)

with open(SCRIPT_DIR + "/../tools/filelists/ss_large_all.txt", "w+") as f:
    for _, path in ss_loader:
        partial_path = "ss" + "ss".join(path.split('ss')[1:])
        f.write(partial_path + "\n")
