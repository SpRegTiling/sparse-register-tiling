import numpy as np
import torch
import pandas as pd
import math
import xformers
import matplotlib.pyplot as plt
import seaborn as sns
import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

#from spmm_benchmarks.loaders.suitesparse import SuiteSparseLoader
from sbench.loaders.dlmc import DLMCLoader
from sbench.loaders.load import load_dense, load_csr, load_coo
from sbench.DDT import spmx_trace_gen, compute_consecutive_fopds, hist_mine_consecutive_fopds
from sbench.utils.cache import cache_dataframe, cached_return


PLOT_DIR = SCRIPT_DIR + "/../plots/"
dlmc_loader = DLMCLoader(file_list=SCRIPT_DIR + "/../tools/matrices_to_test_dlmc_only.txt", loader=load_dense)

torch.set_grad_enabled(False)


tile_shape = (32, 32)


def tile_matrix(mtx, tile_shape):
    return [(i, j, tile)
            for i, x in enumerate(torch.split(mtx, tile_shape[0], 0))
            for j, tile in enumerate(torch.split(x, tile_shape[1], 1))]

def density_sort(_tile):
    col_permutation = torch.argsort(_tile.sum(0))
    _tile = torch.index_select(_tile, 1, col_permutation)

    row_permutation = torch.argsort(_tile.sum(1))
    _tile = torch.index_select(_tile, 0, row_permutation)

    return _tile


dense_threshold = 0.35
plot_dense_tiles = False
results = []
for matrix, path in dlmc_loader:
    A_nnz = matrix.sum().item()
    matrix = density_sort(matrix)

    for tile_shape in [(16, 16)]:
        print(matrix.shape)
        print(path)

        H_norm = 0
        ell_padding_required = matrix.sum(1).max() * matrix.shape[0] - matrix.sum()

        row_counts = matrix.sum(1).numpy()
        col_counts = matrix.sum(0).numpy()

        n_nnzrow_vertical_strips = [0] * math.ceil(matrix.shape[0] / tile_shape[0])
        nnz_per_tile = []
        col_ovl_tile = []
        num_dense_tile = 0
        dense_vs_sparse = 0
        tile_mat_shape = (int(matrix.shape[0]/tile_shape[0]), int(matrix.shape[1]/tile_shape[1]))
        tile_pattern_mat = np.zeros(tile_mat_shape)
        tile_ell_padding_required = 0

        for ti, tj, tile in tile_matrix(matrix, tile_shape):

            nnz_per_tile.append(tile.sum().item())
            density = tile.sum().item() / np.prod(tile_shape)
            if density > dense_threshold:
                tile_pattern_mat[ti][tj] = 1
            tile_ell_padding_required += tile.sum(1).max().item() * tile_shape[0] - tile.sum().item()

            loads = tile.sum(0).bool().sum().item()
            nnz = tile.sum().item()
            if nnz == 0:
                col_ovl_tile.append(0)
            else:
                col_ovl_tile.append((nnz - loads) / nnz)

            n_nnzrow_vertical_strips[ti] += tile.sum(1).bool().sum().item()
            H_norm -= torch.nan_to_num(tile.sum(1) / A_nnz * torch.log(tile.sum(1) / A_nnz)).sum() * 1 / math.log(A_nnz)

        vol_tile = np.prod(tile_shape)
        nnz_pct_per_tile = np.array(nnz_per_tile) / vol_tile
        # sum of nnz in dense tile with respect to the threshold
        dense_vs_sparse = sum(np.array(nnz_per_tile)[np.where(
            nnz_pct_per_tile > dense_threshold)[0].tolist()]) / A_nnz
        num_dense_tile = len(np.where(nnz_pct_per_tile > dense_threshold)[0])
        if plot_dense_tiles:
            plt.spy(tile_pattern_mat)
            plt.gcf().set_size_inches(18.5, 18.5)
            plt.show()

        col_ovl_tile = np.array(col_ovl_tile)


        nnz = matrix.sum().item()
        print(
            (np.array(n_nnzrow_vertical_strips).mean() / matrix.shape[0]),
            (A_nnz / matrix.shape[0]),
            (1 - H_norm)
        )
        ssf = (np.array(n_nnzrow_vertical_strips).mean() / matrix.shape[0]) * (A_nnz / matrix.shape[0]) * (1 - H_norm)
        print(ssf)

        matrix_path = "dlmc" + path.split("dlmc")[-1]
        _, model, pruning, sparsity = matrix_path.split("/")[:4]

        results.append({
            "matrixPath": matrix_path,
            "model": model,
            "pruning": pruning,
            "sparsity": sparsity,
            "m": matrix.shape[0],
            "k": matrix.shape[1],
            "nnz": nnz,
            "SSF": ssf.item(),
            "H_norm": H_norm.item(),
            "H_norm_normalized":  ((A_nnz / matrix.shape[0]) * (1 - H_norm)).item(),
            "row cov": np.std(row_counts) / np.mean(row_counts),
            "col cov": np.std(col_counts) / np.mean(col_counts),
            "tile shape": "x".join(str(x) for x in tile_shape),
            "tile vol": np.prod(tile_shape),
            "tile nnz pct cov": np.std(nnz_pct_per_tile) / nnz_pct_per_tile.mean(),
            "tile nnz pct min": nnz_pct_per_tile.min(),
            "tile nnz pct max": nnz_pct_per_tile.max(),
            "tile nnz pct mean": nnz_pct_per_tile.mean(),
            "tile col ovl cov": np.var(col_ovl_tile) / col_ovl_tile.mean(),
            "tile col ovl min": col_ovl_tile.min(),
            "tile col ovl max": col_ovl_tile.max(),
            "tile col ovl mean": col_ovl_tile.mean(),
            "tile ell padding required": tile_ell_padding_required / nnz,
            "ell padding required": ell_padding_required.item() / nnz,
            "dense computation ratio": dense_vs_sparse,
            "number of dense tiles": num_dense_tile
        })

df = pd.DataFrame(results)
print(df)

df.to_csv(f'{SCRIPT_DIR}/../results/matrix_properties_ord32.csv')
