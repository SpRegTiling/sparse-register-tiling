import numpy
import numpy as np
import torch
import pandas as pd
import math
import scipy
from scipy.io import mmwrite
import xformers
import matplotlib.pyplot as plt
import seaborn as sns
import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
import time, sys


#from spmm_benchmarks.loaders.suitesparse import SuiteSparseLoader
from sbench.loaders.dlmc import DLMCLoader
from sbench.loaders.load import load_dense, load_csr, load_coo

import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

from padding_reordering import find_max_rectangle_list, \
    get_codelet_decomposition, density_sort, tile_matrix, tiling_info, \
    dense_threshold, plot_dense_tiles, get_tiled_codelet_decomposition, \
    export_to_file, convert_tiled_to_matrix_cmp, find_ratio_rectangle_list, \
    PACK_TYPE

PLOT_DIR = SCRIPT_DIR + "/../plots/"

dlmc_loader = DLMCLoader(file_list=SCRIPT_DIR + "/../tools/filelists/rn50_magnitude_70_80.txt", loader=load_dense)
#dlmc_loader = DLMCLoader(file_list=SCRIPT_DIR +  "/../tools/filelists/transformer_magnitude_70_80.txt", loader=load_dense)
# dlmc_loader = DLMCLoader(file_list=SCRIPT_DIR +"/../tools/matrices_to_test_dlmc_only.txt", loader=load_dense)
#dlmc_loader = DLMCLoader(file_list=SCRIPT_DIR +
# "/../tools/filelists/transformer_magnitude_70_80.txt", loader=load_dense)
codelet_out_file = f'{SCRIPT_DIR}/../results/codelets/'

torch.set_grad_enabled(False)


pack_strategy = PACK_TYPE.AGG_ROW

def export_codelets_to_file(codelet_list, out_path):
    f = open(out_path+".hyb_tiles", "w")
    size_cl = 0
    for cl in codelet_list:
        size_cl += len(cl)
    f.write(str(size_cl) + "\n")
    for cll in codelet_list:
        for cl in cll:
            if cl.d_or_sp == 1:
                f.write("d ")
            else:
                f.write("s ")

            f.write(str(cl.x) + " " + str(cl.y) + " ")
            f.write(str(cl.x_offset) + " " + str(cl.y_offset) + " \n")
    f.close()



def gemm_blk(m, c_list, B, ti):
    time_blks = 0
    C = torch.zeros(m, B.shape[1], dtype=torch.float32)
    tile_k = ti["k tile"]
    for j in range(len(c_list)):
        for k in range(int(B.shape[1]/tile_k)):
            for rb in c_list[j]:
                if not rb.d_or_sp:
                    continue
                bb = B[rb.y:rb.y+rb.y_offset, k*tile_k:(k+1)*tile_k]
                tic = time.perf_counter()
                cb = rb.matrix @ bb
                toc = time.perf_counter()
                time_blks += (toc - tic)
                C[rb.x:rb.x + rb.x_offset, k * tile_k:(k + 1) * tile_k] = cb
    return C, time_blks

# Separated Executor, TODO: you may add another separated variant using
#  Full-blcoked approach, like no tiling on B!
def hybrid_spmm(sp_mat, c_list, B, ti):
    C, time_dense = gemm_blk(sp_mat.shape[0], c_list, B, ti)
    #ridx = sp_mat.crow_indices().to(dtype=torch.int)
    #cidx = sp_mat.col_indices().to(type=torch.int)
    tic = time.perf_counter()
    if B.shape[1] == 256 and False:
        Ctmp = torch.ops.sparse.spmm_csr_c_1d_256(sp_mat.values(),
                                            sp_mat.crow_indices(),
                                       sp_mat.col_indices(), B)
        # Ctmp = torch.ops.sparse.spmm_csr_c_1d_256(sp_mat.values(), ridx,
        #                                           cidx, B)
    elif B.shape[1] == 512 and False:
        Ctmp = torch.ops.sparse.spmm_csr_c_1d_512(sp_mat.values(),
                                                  sp_mat.crow_indices(),
                                                  sp_mat.col_indices(), B)
    else:
        Ctmp = sp_mat @ B
    toc = time.perf_counter()
    C = C + Ctmp
    time_sparse = toc - tic
    return C, time_dense, time_sparse

# Blocked Executor
def hybrid_spmm_combined(sp_mat, c_list, B, ti):
    [time_d_blks, time_s_blks] = [0, 0]
    m = sp_mat.shape[0]
    C = torch.zeros(m, B.shape[1], dtype=torch.float32)
    tile_k = ti["k tile"]
    for j in range(len(c_list)):
        dense_visted = False
        for k in range(int(B.shape[1]/tile_k)):
            for rb in c_list[j]:
                if not rb.d_or_sp:
                    bb = B[rb.y:rb.y+rb.y_offset, k*tile_k:(k+1)*tile_k]
                    tic = time.perf_counter()
                    cb = rb.matrix @ bb
                    toc = time.perf_counter()
                    time_s_blks += (toc - tic)
                    C[rb.x:rb.x + rb.x_offset, k * tile_k:(k + 1) * tile_k] \
                        += cb
                elif not dense_visted:
                    bb = B[rb.y:rb.y + rb.y_offset, :]
                    tic = time.perf_counter()
                    cb = rb.matrix @ bb
                    toc = time.perf_counter()
                    time_d_blks += (toc - tic)
                    dense_visted = True
                    C[rb.x:rb.x + rb.x_offset, :] += cb
    return C, time_d_blks, time_s_blks

# Full-Blocked Executor
def hybrid_spmm_combined_fullblocked(sp_mat, c_list, B, ti):
    [time_d_blks, time_s_blks] = [0, 0]
    m = sp_mat.shape[0]
    C = torch.zeros(m, B.shape[1], dtype=torch.float32)
    tile_k = ti["k tile"]
    for j in range(len(c_list)):
        dense_visted = False
        for k in range(int(B.shape[1]/tile_k)):
            for rb in c_list[j]:
                if not rb.d_or_sp:
                    bb = B[rb.y:rb.y+rb.y_offset, k*tile_k:(k+1)*tile_k]
                    tic = time.perf_counter()
                    cb = rb.matrix @ bb
                    toc = time.perf_counter()
                    time_s_blks += (toc - tic)
                    C[rb.x:rb.x + rb.x_offset, k * tile_k:(k + 1) * tile_k] \
                        += cb
                else:
                    bb = B[rb.y:rb.y+rb.y_offset, k*tile_k:(k+1)*tile_k]
                    tic = time.perf_counter()
                    cb = rb.matrix @ bb
                    toc = time.perf_counter()
                    time_d_blks += (toc - tic)
                    dense_visted = True
                    C[rb.x:rb.x + rb.x_offset, k * tile_k:(k + 1) * tile_k] += cb
    return C, time_d_blks, time_s_blks


def spmm(mat_sparse, B):
    tic = time.perf_counter()
    C = mat_sparse @ B
    toc = time.perf_counter()
    return C, toc-tic


def gemm(mat_sparse_dense, B):
    tic = time.perf_counter()
    C = mat_sparse_dense@B
    toc = time.perf_counter()
    return C, toc-tic

def isPowerOfTwo(n):
    return (math.ceil(math.log(n,2)) == math.floor(math.log(n,2)));

import ctypes


results = []
blk_thr=0.3
def run_gemm_hybrid(tile_size, bcol_size, out_path, density, blk_agg):
    tiling_info["m tile"] = tile_size
    tiling_info["n tile"] = tile_size
    tiling_info["k tile"] = tile_size
    tiling_info["bCol"] = bcol_size
    dense_threshold = density
    mkl_rt = ctypes.CDLL('libmkl_rt.so')
    mkl_get_max_threads = mkl_rt.mkl_get_max_threads
    def mkl_set_num_threads(cores):
        mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(cores)))
    mkl_set_num_threads(1)
    torch.set_num_threads(1)


    for matrix, path in dlmc_loader:
        A_nnz = matrix.sum().item()
        matrix = density_sort(matrix)
        if not isPowerOfTwo(matrix.shape[0]) or not isPowerOfTwo(matrix.shape[1]):
            continue
        # scipy.random.seed(seed=33)
        # A_spm_mat = scipy.sparse.random(16, 32, 0.3,
        #                                 dtype=np.float32).toarray()
        # torch_csr = torch.tensor(A_spm_mat).to_sparse_csr()
        # matrix = torch_csr.to_dense()
        # matrix = density_sort(matrix)

        # csr_mtx = scipy.sparse.csr_matrix(matrix.numpy())
        # name = "/home/kazem/development/dnn-spmm-bench/results/d.mtx"
        # mmwrite(name, csr_mtx)

        for tile_shape in [(tiling_info['m tile'], tiling_info['n tile'])]:
            #print(matrix.shape)
            #print(path)
            if matrix.shape[0] <= tile_shape[0] or matrix.shape[1] <= \
                tile_shape[1]:
                continue
            H_norm = 0
            ell_padding_required = matrix.sum(1).max() * matrix.shape[0] - matrix.sum()

            row_counts = matrix.sum(1).numpy()
            col_counts = matrix.sum(0).numpy()

            n_nnzrow_vertical_strips = [0] * math.ceil(matrix.shape[0] / tile_shape[0])
            nnz_per_tile = []
            paddedhybrid = 0
            col_ovl_tile = []
            num_dense_tile = 0
            dense_vs_sparse = 0
            tile_mat_shape = (int(matrix.shape[0]/tile_shape[0]), int(matrix.shape[1]/tile_shape[1]))
            tile_pattern_mat = np.zeros(tile_mat_shape)
            tile_nnz = np.zeros(tile_mat_shape)
            tile_ell_padding_required = 0

            for ti, tj, tile in tile_matrix(matrix, tile_shape):
                nnz_per_tile.append(tile.sum().item())
                tile_nnz[ti,tj] = tile.sum().item()
                density = tile.sum().item() / np.prod(tile_shape)
                if density > dense_threshold:
                    tile_pattern_mat[ti][tj] = 1
                    paddedhybrid += (np.prod(tile_shape) - tile.sum().item())
                tile_ell_padding_required += tile.sum(1).max().item() * tile_shape[0] - tile.sum().item()
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
            if pack_strategy == PACK_TYPE.AGG_ROW or pack_strategy == \
                PACK_TYPE.ROW_BASED:
                rct_max_list = find_max_rectangle_list(tile_pattern_mat,
                                                       pack_strategy, blk_agg,10)
            else:
                rct_max_list = find_ratio_rectangle_list(tile_pattern_mat,
                                                     matrix, A_nnz,
                                                     tile_nnz, blk_thr)
            c_list, sp_mat = get_tiled_codelet_decomposition(matrix, tile_shape,
                                                    rct_max_list)
            c_let_file_name = os.path.basename(path).split(".")[0]
            c_let_file_name += ("_"+str(int(dense_threshold*100))+"_"+str(
                agg)+"_"+str(tile_size))
            if export_to_file:
                export_codelets_to_file(c_list, codelet_out_file+c_let_file_name)
                mat_csr = scipy.sparse.csr_matrix(matrix)
                mmwrite(codelet_out_file + c_let_file_name + ".mtx", mat_csr)
                f = open(codelet_out_file + "file_list.txt", "a")
                f.write("    - dlmc/transformer/magnitude_pruning/0.7/" +
                        c_let_file_name + ".mtx\n")
                f.close()

            Bmat = torch.ones(matrix.shape[1], tiling_info["bCol"],
                          dtype=torch.float32)
            #matrix_csr = scipy.sparse.csr_matrix(matrix)
            orig_matrix = matrix.clone().detach()
            torch_csr = torch.tensor(orig_matrix).to_sparse_csr()
            torch_csr = torch.sparse_csr_tensor(
                torch_csr.crow_indices().to(dtype=torch.int),
                torch_csr.col_indices().to(dtype=torch.int),
                torch_csr.values(),
                size=torch_csr.shape)
            convert_tiled_to_matrix_cmp(matrix.shape[0],matrix.shape[1],
                                        c_list, orig_matrix)

            num_test = 9
            t_array = []
            for i in range(num_test):
                C_gemm, tg = gemm(matrix, Bmat)
                t_array.append(tg)
            t_gemm = np.median(t_array)
            C_gemm, tg = gemm(matrix, Bmat)

            t_array = []
            for i in range(num_test):
                C_spmm, ts = spmm(torch_csr, Bmat)
                t_array.append(ts)
                if torch.max(abs(C_gemm-C_spmm)) > 1e-4:
                    print(" SpMM NOT EQUAL! ")
            t_spmm = np.median(t_array)


            t_array_h = []
            for i in range(num_test):
                C_hybrid, hybrid_dense, hybrd_sparse = hybrid_spmm_combined(sp_mat, c_list,
                                                               Bmat, tiling_info)
                t_array_h.append((hybrid_dense, hybrd_sparse,
                                  hybrid_dense+hybrd_sparse))
                if torch.max(abs(C_gemm-C_hybrid)) > 1e-4:
                    print(" Hybrid NOT EQUAL! ")
            hybrid_time = np.median([x[2] for x in t_array_h])
            hybrid_sparse = t_array_h[4][1]
            hybrid_dense = t_array_h[4][0]

            ### Hybrid 2
            t_array_h = []
            for i in range(num_test):
                C_hybrid, hybrid_dense, hybrd_sparse = hybrid_spmm(sp_mat, c_list,
                                                               Bmat, tiling_info)
                t_array_h.append((hybrid_dense, hybrd_sparse,
                                  hybrid_dense+hybrd_sparse))
                if torch.max(abs(C_gemm-C_hybrid)) > 1e-4:
                    print(" Hybrid SEP NOT EQUAL! ")
            hybrid_time_sep = np.median([x[2] for x in t_array_h])
            hybrid_sparse_sep = t_array_h[4][1]
            hybrid_dense_sep = t_array_h[4][0]


            ## Hybrid 3
            t_array_h = []
            for i in range(num_test):
                C_hybrid, hybrid_dense, hybrd_sparse = hybrid_spmm_combined_fullblocked(sp_mat,
                                                                    c_list,
                                                               Bmat, tiling_info)
                t_array_h.append((hybrid_dense, hybrd_sparse,
                                  hybrid_dense+hybrd_sparse))
                if torch.max(abs(C_gemm-C_hybrid)) > 1e-4:
                    print(" Hybrid Full Blocked NOT EQUAL! ")
            hybrid_time_fb = np.median([x[2] for x in t_array_h])
            hybrid_sparse_fb = t_array_h[4][1]
            hybrid_dense_fb = t_array_h[4][0]


            max_su = np.min([t_gemm, t_spmm]) / np.min([hybrid_time, hybrid_time_sep, hybrid_time_fb])
            if max_su >= 1.1 :
                print( max_su, " ", t_gemm, " ",
                       tile_shape, " ", dense_threshold, " ", agg, " ",
                       bcol_size, " ", pack_strategy, " ", path)

            results.append({
                "matrixPath": path,
                "m": matrix.shape[0],
                "k": matrix.shape[1],
                "nnz": A_nnz,
                "hybrid padding": paddedhybrid,
                "number of threads": mkl_get_max_threads,
                "density threshold": dense_threshold,
                "block coverage threshold": blk_thr,
                "packing method": pack_strategy,
                "m tile size": tiling_info["m tile"],
                "n tile size": tiling_info["n tile"],
                "k tile size": tiling_info["k tile"],
                "B Col": tiling_info["bCol"],
                "tile shape": "x".join(str(x) for x in tile_shape),
                "tile vol": np.prod(tile_shape),
                "tile nnz pct min": nnz_pct_per_tile.min(),
                "tile nnz pct max": nnz_pct_per_tile.max(),
                "tile nnz pct mean": nnz_pct_per_tile.mean(),
                "dense computation ratio": dense_vs_sparse,
                "number of dense tiles": num_dense_tile,
                "PyTorch SpMM Time (sec)": t_spmm,
                "PyTorch GEMM Time (sec)": t_gemm,
                "Hybrid Sparse Blocked Time (sec)": hybrid_sparse,
                "Hybrid Dense Blocked Time (sec)": hybrid_dense,
                "Total Hybrid Blocked Time (sec)": hybrid_time,
                "Hybrid Sparse Sep Time (sec)": hybrid_sparse_sep,
                "Hybrid Dense Sep Time (sec)": hybrid_dense_sep,
                "Total Hybrid Sep Time (sec)": hybrid_time_sep,
                "Hybrid Sparse Full-Blocked Time (sec)": hybrid_sparse_fb,
                "Hybrid Dense Full-Blocked Time (sec)": hybrid_dense_fb,
                "Total Hybrid Full-Blocked Time (sec)": hybrid_time_fb
            })


            #print(hybrid_dense, ",", hybrd_sparse, ",", t_spmm, ",", t_gemm)

if __name__ == "__main__":
    args = sys.argv[1:]
    # tile_size = int(args[0])
    # bcol_size = int(args[1])
    # out_path = args[2]
    # density = float(int(args[3])/100)
    out_path= f'{SCRIPT_DIR}/../results/hybrid_70_80_agg_MKL_resnet.csv'
    pack_strategy = PACK_TYPE.PERCENTAGE
    for density in (0.9, 0.8, 0.7, 0.6, 0.5, 0.55, 0.4, 0.45, 0.3, 0.2, 0.1):
        blk_thr = density
        agg=2
        for tile_size in (32, 64, 128):
            for bcol_size in {64, 128, 256}:
                if tile_size > bcol_size:
                    continue
                run_gemm_hybrid(tile_size, bcol_size, out_path, density, agg)

    pack_strategy = PACK_TYPE.ROW_BASED
    for density in (0.2, 0.23, 0.25, 0.28, 0.3, 0.33, 0.35 ):
        for tile_size in (32, 64, 128):
            for bcol_size in {64, 128, 256}:
                if tile_size > bcol_size:
                    continue
                run_gemm_hybrid(tile_size, bcol_size, out_path, density, agg)

    pack_strategy = PACK_TYPE.AGG_ROW
    for density in (0.2, 0.23, 0.25, 0.28, 0.3, 0.33, 0.35 ):
        for agg in (2, 3, 4, 5, 10, 20, 30):# (2, 3, 4, 10, 20):
            for tile_size in (32, 64, 128):
                for bcol_size in {64, 128, 256}:
                    if tile_size > bcol_size:
                        continue
                    run_gemm_hybrid(tile_size, bcol_size, out_path, density, agg)

    df = pd.DataFrame(results)
    df.to_csv(out_path)
