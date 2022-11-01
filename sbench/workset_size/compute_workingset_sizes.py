import numpy as np
import torch
import pandas as pd
import math
import sbench
import random
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from functools import partial
from itertools import islice
from scipy.stats import variation
import pickle
import ssgetpy

from typing import List
import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

#from spmm_benchmarks.loaders.suitesparse import SuiteSparseLoader
from sbench.loaders.dlmc import DLMCLoader
from sbench.loaders.suitesparse import SuiteSparseLoader
from sbench.loaders.load import load_dense, load_csr, load_coo
from sbench.DDT import spmx_trace_gen, compute_consecutive_fopds, hist_mine_consecutive_fopds
from sbench.utils.cache import cache_dataframe, cached_return


CACHE_DIR = "/sdb/cache/workingset_size/"
os.makedirs(CACHE_DIR, exist_ok=True)

SS_CACHE_DIR = CACHE_DIR + "/ss"
os.makedirs(SS_CACHE_DIR, exist_ok=True)

DLMC_CACHE_DIR = CACHE_DIR + "/dlmc"
os.makedirs(DLMC_CACHE_DIR, exist_ok=True)

print(DLMC_CACHE_DIR)

BCOLS = 256

if __name__ == "__main__":
    random.seed(10)
    SS_MATRICES_TO_PLOT = 507
    SS_MATRICES_TO_SELECT_FROM = 5000
    # Matrices that consume too much ram for our current setup
    SS_SKIP_LIST = ["wikipedia-20051105", "circuit5M_dc", "memchip", "mycielskian19", "mycielskian20", "333SP", "ss"]
    MAX_SS_MATRIX_SIZE = 4e6

    matrix_list = ssgetpy.search(rowbounds=[0, MAX_SS_MATRIX_SIZE],
                                 colbounds=[0, MAX_SS_MATRIX_SIZE],
                                 limit=SS_MATRICES_TO_SELECT_FROM)
    matrix_ids = [matrix.id for matrix in matrix_list]
    random.shuffle(matrix_ids)

    PLOT_DIR = SCRIPT_DIR + "/../plots/"

    torch.set_grad_enabled(False)
    tile_shapes = [(x, x) for x in range(2, 512, 4)]

    #
    # BUCKET_SIZE = 0.05
    # num_buckets = int(1 / BUCKET_SIZE) + 1

    def run(loader, name, cache_dir, recompute=False):
        variation_per_tilesize = []
        tile_sizes = []
        matrix_density = []

        tile_sizes = np.array([x[0] for x in tile_shapes])
        np.save(cache_dir + "/tile_sizes.npy", tile_sizes)

        for matrix, path in loader:
            print(path)
            filepath = "/".join(path.split("/")[4:-1])
            filename = os.path.basename(path).split('.')[0]
            nnz = matrix.to(torch.bool).sum().sum().item()
            density = nnz / (matrix.shape[0] * matrix.shape[1])
            sparsity = round(1 - density, 2)
            print(filepath, sparsity, nnz, matrix.shape,end=' ')

            cache_dir_tmp = cache_dir + f"/{filepath}"
            os.makedirs(cache_dir_tmp, exist_ok=True)

            if os.path.exists(cache_dir_tmp + f"/{filename}_working_set_sizes.npy"):
                print("skipped")
                continue
            else:
                print("computing")

            for tile_shape in tile_shapes:
                (tile_rows, tile_cols) = tile_shape

                num_tiles, num_empty_tiles, active_rows, active_cols, densities = \
                    torch.ops.spmm_benchmarks.tile_stats_csr_not_binned(tile_rows, tile_cols, matrix)

                assert num_empty_tiles >= 0 and num_empty_tiles <= num_tiles

                active_rows = active_rows.numpy()
                active_cols = active_cols.numpy()
                densities = densities.numpy()

                # Assume CSR storage
                nnz = densities * tile_rows * tile_cols
                working_set_sizes = nnz * 2 + tile_rows \
                                    + BCOLS * active_rows * tile_rows \
                                    + BCOLS * active_cols * tile_cols

                np.save(cache_dir_tmp + f"/{filename}_working_set_sizes.npy", working_set_sizes)
                np.save(cache_dir_tmp + f"/{filename}_active_rows.npy", active_rows)
                np.save(cache_dir_tmp + f"/{filename}_active_cols.npy", active_cols)
                np.save(cache_dir_tmp + f"/{filename}_densities.npy", densities)

                del active_rows, active_cols, densities, working_set_sizes
            del matrix


    # dlmc_loader = islice(DLMCLoader(loader=load_csr, models=["transformer"],
    #                                 pruning_methods=["magnitude_pruning"], sparsities=[0.7]), 1)
    dlmc_loader = DLMCLoader(loader=load_csr)
    run(dlmc_loader, "ml", cache_dir=DLMC_CACHE_DIR, recompute=False)

    # ss_loader = islice(SuiteSparseLoader(matrix_ids=matrix_ids[:SS_MATRICES_TO_PLOT],
    #                                      skip_list=SS_SKIP_LIST, loader=load_csr), 1)
    ss_loader = SuiteSparseLoader(matrix_ids=matrix_ids[:SS_MATRICES_TO_PLOT], skip_list=SS_SKIP_LIST, loader=load_csr)
    run(ss_loader, "ss", cache_dir=SS_CACHE_DIR, recompute=False)

