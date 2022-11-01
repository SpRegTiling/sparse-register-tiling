import numpy as np
import torch
import pandas as pd
import math
import scipy
import sys
from scipy.io import mmwrite
import xformers
import matplotlib.pyplot as plt
import seaborn as sns
import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
from enum import Enum


#from spmm_benchmarks.loaders.suitesparse import SuiteSparseLoader
from sbench.loaders.dlmc import DLMCLoader
from sbench.loaders.load import load_dense, load_csr, load_coo
from sbench.DDT import spmx_trace_gen, compute_consecutive_fopds, hist_mine_consecutive_fopds
from sbench.utils.cache import cache_dataframe, cached_return


PLOT_DIR = SCRIPT_DIR + "/../plots/"
dlmc_loader = DLMCLoader(file_list=SCRIPT_DIR + "/../tools/matrices_to_test_dlmc_only.txt", loader=load_dense)

torch.set_grad_enabled(False)


class PACK_TYPE(Enum):
    ROW_BASED = 1 # each row tile is a blocked starting from the first dense
    # tile till the end
    AGG_ROW = 2 # every agg consecutive row tile will be considered for blccking
    PERCENTAGE = 3 # starting from the below right corener, it adds tiles to
    # the block to meet cover x percentage of nonzeros. This naive and does
    # not check density of tiles.

dense_threshold = 0.30
plot_dense_tiles = False
export_to_file = False
tiling_info = {"bCol": 128, "m tile": 64, "n tile": 64, "k tile": 64}


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


class rectangle:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height


def get_max_rectangle(rect_list):
    max_area = 0
    max_rect = -1
    for r in range(len(rect_list)):
        if rect_list[r].width*rect_list[r].height > max_area:
            max_rect = r
    return rect_list[max_rect]


def find_row_block(in_bit_map, x, y):
    ret_rct = None
    for i in range(x, in_bit_map.shape[0]):
        if np.sum(in_bit_map[i]) == 0:
            continue
        loc_beg = np.min(np.where(in_bit_map[i] == 1))
        ret_rct = rectangle(i, loc_beg, in_bit_map.shape[1]-loc_beg, 1)
        return [ret_rct]


def find_agg_row_block(in_bit_map, x, y, agg, thr):
    [loc_beg, aggu] = [0, 0]
    bnd = np.min([x+agg, in_bit_map.shape[0]])
    for i in range(x, in_bit_map.shape[0]):
        if np.sum(in_bit_map[i]) == 0:
            continue
        for j in range(i, bnd):
            if np.sum(in_bit_map[j]) == 0:
                break
            loc_beg = np.max([np.min(np.where(in_bit_map[j] == 1)), loc_beg])
            aggu = aggu+1
        ret_rct = rectangle(i, loc_beg, in_bit_map.shape[1]-loc_beg, aggu)
        return [ret_rct]


def set_row_tile(in_bit_map, rct_lst):
    for rct in rct_lst:
        for i in range(rct.x, rct.x+rct.height):
            in_bit_map[i][:] = 0
    return in_bit_map


def find_max_rectangle_from_right_corner(in_bit_map, x, y):
    wdt = x
    hgt = y
    for i in range(x, in_bit_map.shape[0]):
        for j in range(y, in_bit_map.shape[1]):
            rows = in_bit_map[i][x:i]
            cols = in_bit_map[j][y:j]
            if len(rows) != np.sum(rows):
                sq_wdt = wdt
            if len(cols) != np.sum(cols):
                sq_hgt = hgt
    while in_bit_map[wdt][hgt]:
        wdt = wdt-1
        hgt = hgt-1
    rct_sq = rectangle(x, y, abs(x-wdt), abs(y-hgt))
    sq_wdt = wdt
    sq_hgt = hgt
    while in_bit_map[sq_wdt][hgt]:
        hgt = hgt-1
    rct_skinny = rectangle(x, y, abs(sq_wdt - x), abs(hgt - y))
    while in_bit_map[wdt][sq_hgt]:
        wdt = wdt-1
    rct_fat = rectangle(x, y, abs(wdt - x), abs(sq_hgt - y))
    return get_max_rectangle([rct_sq, rct_skinny, rct_fat])


def find_next_available_corner(in_bit_map):
    for cur_i in range(in_bit_map.shape[0]):
        for cur_j in range(in_bit_map.shape[1]):
            if in_bit_map[cur_i][cur_j]:
                return cur_i, cur_j
    return -1, -1


def set_bit_map(in_bit_map, rct_list):
    for rct in rct_list:
        for i in range(rct.x, rct.x+rct.height ):
            for j in range(rct.y, rct.y+rct.width):
                in_bit_map[i][j] = 0
    return in_bit_map


def find_max_rectangle_list(in_bit_map_,pack_strategy, blk_agg=4,
                            blk_threshold=10):
    final_list = []
    all_visited = np.sum(np.sum(in_bit_map_))
    in_bit_map = np.copy(in_bit_map_)
    aggregation = False
    if pack_strategy == PACK_TYPE.AGG_ROW:
        aggregation = True
    while all_visited > 0:
        ci, cj = find_next_available_corner(in_bit_map)
        if aggregation:
            max_rct = find_agg_row_block(in_bit_map, ci, cj, blk_agg,
                                         blk_threshold)
        else:
            max_rct = find_row_block(in_bit_map, ci, cj)
        in_bit_map = set_bit_map(in_bit_map, max_rct)
        in_bit_map = set_row_tile(in_bit_map, max_rct)
        all_visited = np.sum(np.sum(in_bit_map))
        final_list.append(max_rct)
    return final_list


def find_ratio_rectangle_list(in_bit_map_, original_matrix, A_nnz, tile_nnz,
                              ratio=0.2):
    final_list = []
    all_visited = np.sum(np.sum(in_bit_map_))
    in_bit_map = np.copy(in_bit_map_)
    aggregation = True
    origi, origj = [in_bit_map.shape[0], in_bit_map.shape[1]]
    ci, cj = [origi, origj]
    target_nnz = ratio * A_nnz
    coverd_nnz = 0
    while coverd_nnz < target_nnz:
        coverd_nnz = 0
        if ci-1>0:
            ci = ci-1
        if cj-1>0:
            cj = cj-1
        for i in range(ci, origi, 1):
            for j in range(cj, origj, 1):
                coverd_nnz += tile_nnz[i][j]
        if ci == 1 and cj == 1:
            break
    for i in range(ci, origi, 1):
        for j in range(cj, origj, 1):
            in_bit_map[i][j] = 0
    rct_fat = rectangle(ci, cj, abs(origj - cj), abs(origi - ci))
    final_list.append([rct_fat])
    return final_list


class Codelet:
    def __init__(self, matrix, x, y, x_offset, y_offset, d_or_sp):
        self.matrix = matrix
        self.x = x
        self.y = y
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.d_or_sp = d_or_sp


def get_codelet_decomposition(original_matrix, tile_shape, rct_2D_list):
    codelet_list = []
    sparse_mat = original_matrix.clone().detach()
    for lst in rct_2D_list:
        clr = []
        for r in lst:
            x1 = r.x*tile_shape[0]
            x_offset = r.height*tile_shape[0]
            y1 = r.y*tile_shape[1]
            y_offset = r.width * tile_shape[1]
            block = sparse_mat[x1:x1+x_offset, y1:y1+y_offset].clone().detach()
            clr.append( Codelet(block, x1, y1, x_offset, y_offset, 1) )
            sparse_mat[x1:x1 + x_offset, y1:y1 + y_offset] = 0
        codelet_list.append(clr)
    #csr_mtx = scipy.sparse.csr_matrix(sparse_mat)
    tt_dense = torch.tensor(sparse_mat).clone().detach()
    torch_csr = tt_dense.to_sparse_csr()
    torch_csr = torch.sparse_csr_tensor(
        torch_csr.crow_indices().to(dtype=torch.int),
        torch_csr.col_indices().to(dtype=torch.int),
        torch_csr.values(),
        size=torch_csr.shape)
    if plot_dense_tiles:
        plt.spy(sparse_mat)
        plt.gcf().set_size_inches(18.5, 18.5)
        plt.show()
    return codelet_list, torch_csr


def convert_tiled_to_matrix_cmp(m, n, c_list, orig):
    mat = torch.zeros((m,n), dtype=torch.float32)
    for cl in c_list:
        for c in cl:
            mat[c.x:c.x+c.x_offset, c.y:c.y+c.y_offset] = c.matrix.to_dense()\
                if not c.d_or_sp else c.matrix
    cmp = torch.eq(mat, orig)
    if not mat.shape[0] * mat.shape[1] == cmp.sum().sum():
        print(" A NOT EQUAL! ")
    return mat, cmp

def get_tiled_codelet_decomposition(original_matrix, tile_shape, rct_2D_list):
    codelet_list = []
    sparse_mat = original_matrix.clone().detach()
    if len(rct_2D_list) == 0: # no dense block is found
        for qq in range(int(original_matrix.shape[0]/tile_shape[0])):
            clr = []
            for q in range(int(original_matrix.shape[1]/tile_shape[1])):
                x1 = qq * tile_shape[0]
                y1 = q * tile_shape[1]
                blk = original_matrix[x1:x1 + tile_shape[0], y1:y1 + tile_shape[
                    1]].clone().detach()
                # blks = scipy.sparse.csr_matrix(blk)
                torch_csr = blk.to_sparse_csr()
                torch_csr = torch.sparse_csr_tensor(
                    torch_csr.crow_indices().to(dtype=torch.int),
                    torch_csr.col_indices().to(dtype=torch.int),
                    torch_csr.values(),
                    size=torch_csr.shape)
                clr.append(
                    Codelet(torch_csr, x1, y1, tile_shape[0], tile_shape[0], 0))
            codelet_list.append(clr)
    rx_prev = 0
    for lst in rct_2D_list:
        clr, r = [], lst[0]
        if r.x > rx_prev:  # beginning tiles
            for ll in range(rx_prev, r.x):
                for mm in range(int(original_matrix.shape[1] / tile_shape[0])):
                    x1 = ll * tile_shape[0]
                    y1 = mm * tile_shape[1]
                    blk = original_matrix[x1:x1 + tile_shape[0],
                          y1:y1 + tile_shape[1]].clone().detach()
                    # blks = scipy.sparse.csr_matrix(blk)
                    # blks = torch.csr_matrix(blk)
                    torch_csr = blk.to_sparse_csr()
                    torch_csr = torch.sparse_csr_tensor(
                        torch_csr.crow_indices().to(dtype=torch.int),
                        torch_csr.col_indices().to(dtype=torch.int),
                        torch_csr.values(),
                        size=torch_csr.shape)
                    clr.append(Codelet(torch_csr, x1, y1, tile_shape[0], tile_shape[0],0))
                codelet_list.append(clr)
                clr = []
        for r in lst:
            for qq in range(r.x, r.x + r.height):# beginning tiles before a dense tile
                for q in range(r.y):
                    x1 = qq * tile_shape[0]
                    y1 = q * tile_shape[1]
                    blk = original_matrix[x1:x1+tile_shape[0], y1:y1+tile_shape[
                        1]].clone().detach()
                    #blks = scipy.sparse.csr_matrix(blk)
                    torch_csr = blk.to_sparse_csr()
                    torch_csr = torch.sparse_csr_tensor(
                        torch_csr.crow_indices().to(dtype=torch.int),
                        torch_csr.col_indices().to(dtype=torch.int),
                        torch_csr.values(),
                        size=torch_csr.shape)
                    clr.append(Codelet(torch_csr, x1, y1, tile_shape[0], tile_shape[0], 0))
            # assuming the rest is one big dense tile
            x1 = r.x * tile_shape[0]
            x_offset = r.height * tile_shape[0]
            y1 = r.y*tile_shape[1]
            y_offset = r.width * tile_shape[1]
            block = sparse_mat[x1:x1+x_offset, y1:y1+y_offset].clone().detach()
            clr.append( Codelet(block, x1, y1, x_offset, y_offset, 1) )
            sparse_mat[x1:x1 + x_offset, y1:y1 + y_offset] = 0
            rx_prev = r.x + r.height
        codelet_list.append(clr)
    #csr_mtx = scipy.sparse.csr_matrix(sparse_mat)
    tt_dense = torch.tensor(sparse_mat).clone().detach()
    torch_csr = tt_dense.to_sparse_csr()
    torch_csr = torch.sparse_csr_tensor(
        torch_csr.crow_indices().to(dtype=torch.int),
        torch_csr.col_indices().to(dtype=torch.int),
        torch_csr.values(),
        size=torch_csr.shape)
    if plot_dense_tiles:
        plt.spy(sparse_mat)
        plt.gcf().set_size_inches(18.5, 18.5)
        plt.show()
    return codelet_list, torch_csr


def run_experiment():
    results = []
    for matrix, path in dlmc_loader:
        A_nnz = matrix.sum().item()
        matrix = density_sort(matrix)
        # csr_mtx = scipy.sparse.csr_matrix(matrix.numpy())
        # name = "/home/kazem/development/dnn-spmm-bench/results/d.mtx"
        # mmwrite(name, csr_mtx)



        for tile_shape in [(tiling_info['m tile'], tiling_info['n tile'])]:
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
            rct_max_list = find_max_rectangle_list(tile_pattern_mat)
            sp_mat, c_list = get_codelet_decomposition(matrix, tile_shape,
                                                    rct_max_list)
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


if __name__ == "__main__":
    args = sys.argv[1:]
    run_experiment()
