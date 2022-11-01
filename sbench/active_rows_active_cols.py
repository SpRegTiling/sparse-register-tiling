import numpy as np
import torch
import pandas as pd
import math
import xformers
import random
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, namedtuple
from dataclasses import dataclass
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


def get_sparsity_range(sparsity):
    sparsity = min(sparsity, 0.9999999999)
    for i, (start, end) in enumerate(sparsity_ranges):
        if start <= sparsity < end:
            return sparsity_ranges[i]
    assert False
    return (-1, -1)


def tile_matrix(mtx, tile_shape):
    return [(i, j, tile)
            for i, x in enumerate(torch.split(mtx, tile_shape[0], 0))
            for j, tile in enumerate(torch.split(x, tile_shape[1], 1))]


@dataclass
class TileStats:
    density_hist: torch.Tensor
    active_cols_hist: torch.Tensor
    active_rows_hist: torch.Tensor
    total_tiles: int
    total_matrices: int
    total_empty_tiles: int


@dataclass
class TileStatsNotBinned:
    active_cols: List[torch.Tensor]
    active_rows: List[torch.Tensor]
    densities: List[torch.Tensor]
    total_matrices: int
    total_tiles: int
    total_empty_tiles: int


# Don't use lambda because it's not pickleables
# def dd2(): return TileStatsNotBinned([], [], [], 0, 0, 0)
# def dd1(): return defaultdict(dd2)

NUM_BINS = 50

# Don't use lambda because it's not pickleables
def dd2(): return TileStats(torch.zeros(NUM_BINS), torch.zeros(NUM_BINS), torch.zeros(NUM_BINS), 0, 0, 0)
def dd1(): return defaultdict(dd2)


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
    tile_shapes = [(128, 128), (64, 64)]

    sparsity_ranges = [(0.7, 0.9), (0.9, 0.95), (0.95, 1.0)]
    #
    # BUCKET_SIZE = 0.05
    # num_buckets = int(1 / BUCKET_SIZE) + 1

    def run(loader, name, bins, recompute=False):
        results = defaultdict(dd1)

        if not os.path.exists(f'active_rows_cols_backup_not_binned_{name}_{bins}.pickle') or recompute:
            for matrix, path in loader:
                print(path)
                nnz = matrix.to(torch.bool).sum().sum().item()
                print(nnz, matrix.shape)
                density = nnz / (matrix.shape[0] * matrix.shape[1])
                sparsity = round(1 - density, 2)
                print(path, sparsity)
                sparsity_range = get_sparsity_range(sparsity)

                for tile_shape in tile_shapes:
                    (tile_rows, tile_cols) = tile_shape

                    num_tiles, num_empty_tiles, active_rows, active_cols, densities = \
                        torch.ops.spmm_benchmarks.tile_stats_csr_not_binned(tile_rows, tile_cols, matrix)

                    assert num_empty_tiles >= 0 and num_empty_tiles <= num_tiles

                    density_hist, _ = np.histogram(densities, range=(0, 1), bins=bins)
                    active_rows_hist, _ = np.histogram(active_rows, range=(0, 1), bins=bins)
                    active_cols_hist, _ = np.histogram(active_cols, range=(0, 1), bins=bins)

                    del active_rows, active_cols, densities

                    print(results[sparsity_range][tile_shape].total_matrices)

                    results[sparsity_range][tile_shape].total_matrices += 1
                    results[sparsity_range][tile_shape].total_tiles += num_tiles
                    results[sparsity_range][tile_shape].total_empty_tiles += num_empty_tiles
                    results[sparsity_range][tile_shape].density_hist += density_hist
                    results[sparsity_range][tile_shape].active_rows_hist += active_rows_hist
                    results[sparsity_range][tile_shape].active_cols_hist += active_cols_hist

                del matrix

            with open(f'active_rows_cols_backup_not_binned_{name}_{bins}.pickle', 'wb') as f:
                pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f'active_rows_cols_backup_not_binned_{name}_{bins}.pickle', 'rb') as f:
            results = pickle.load(f)

        return results

    def plot(results, name, include_empty=False, sparsity_ranges=sparsity_ranges):
        print(name, include_empty)
        for tile_shape in [(64, 64),( 128, 128)]:
            def sparsity_range_str(sparsity_range):
                return f"[{int(sparsity_range[0]*100)}%, {int(sparsity_range[1]*100)}%)"

            total_num_matrices = 0
            data_active_rows = {}
            for sparsity_range in sparsity_ranges:
                total_num_matrices += results[sparsity_range][tile_shape].total_matrices
                tile_active_rows = results[sparsity_range][tile_shape].active_rows_hist.clone().numpy()
                if include_empty:
                    print(results[sparsity_range][tile_shape].total_empty_tiles)
                    tile_active_rows[0] += results[sparsity_range][tile_shape].total_empty_tiles

                data_active_rows[sparsity_range_str(sparsity_range)] = tile_active_rows

            print(data_active_rows)
            # df_rows = pd.concat(pd.DataFrame({'range': k, 'pct_active': v, "active": "rows"}) for k, v in data.items())\
            #           .reset_index()

            data_active_cols = {}
            for sparsity_range in sparsity_ranges:
                tile_active_cols = results[sparsity_range][tile_shape].active_cols_hist.clone().numpy()
                if include_empty:
                    tile_active_cols[0] += results[sparsity_range][tile_shape].total_empty_tiles

                data_active_cols[sparsity_range_str(sparsity_range)] = tile_active_cols

            # df_cols = pd.concat(pd.DataFrame({'range': k, 'pct_active': v, "active": "cols"}) for k, v in data.items())\
            #           .reset_index()
            #df = pd.concat([df_rows, df_cols])

            fig, ax = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

            for sparsity_range in sparsity_ranges:
                # with seaborn (use hist_kws to send arugments to plt.hist, used underneath)
                sns.histplot(x=np.linspace(0, 1, NUM_BINS),
                             bins=NUM_BINS,
                             weights=data_active_rows[sparsity_range_str(sparsity_range)],
                             stat="probability", common_norm=False, element="bars", fill=False, ax=ax[0])
                sns.histplot(x=np.linspace(0, 1, NUM_BINS),
                             bins=NUM_BINS,
                             weights=data_active_cols[sparsity_range_str(sparsity_range)],
                             stat="probability", common_norm=False, element="bars", fill=False, ax=ax[1])

            plt.gcf().suptitle("\n".join(
                [f"{name} ({total_num_matrices} Matrices)",
                 f"(Tile Shape: {tile_shape[0]}x{tile_shape[1]})",
                 '(Excluding Empty Tiles)' if not include_empty else ""]))

            ax[1].legend([sparsity_range_str(sparsity_range) for sparsity_range in sparsity_ranges])

            ax[0].title.set_text('Active Rows')
            ax[1].title.set_text('Active Columns')

            ax[0].set(xlabel="Pct. of Rows that are Active\n(for a given tile)",
                      ylabel=f"Proportion of {'Non-Empty ' if not include_empty else ''}Tiles")
            ax[0].set(xlabel="Pct. of Cols that are Active\n(for a given tile)",
                      ylabel=f"Proportion of {'Non-Empty ' if not include_empty else ''}Tiles")

            plt.savefig(f'active_rows_cols_{name}_{tile_shape[0]}x{tile_shape[1]}_{include_empty}.png')
            plt.clf()
            fig.clf()


    dlmc_loader = DLMCLoader(loader=load_csr)
    results = run(dlmc_loader, "ml", bins=NUM_BINS, recompute=False)
    plot(results, "ML Matrices", include_empty=True)
    plot(results, "ML Matrices", include_empty=False)

    ss_loader = SuiteSparseLoader(matrix_ids=matrix_ids[:SS_MATRICES_TO_PLOT], skip_list=SS_SKIP_LIST, loader=load_csr)
    results = run(ss_loader, "ss", bins=NUM_BINS, recompute=False)
    plot(results, "SuiteSparse Matrices", include_empty=True, sparsity_ranges=sparsity_ranges[1:])
    plot(results, "SuiteSparse Matrices", include_empty=False, sparsity_ranges=sparsity_ranges[1:])
    # df = pd.DataFrame(results)
    # df.to_csv(f'{SCRIPT_DIR}/../results/matrix_properties_ord32.csv')
