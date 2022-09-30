
import scipy.io
import scipy.sparse
import sys, os

from utils import *
from psc_codelet import *
from mosek_test import psc_mining, TSP
from padding import *

from spmm_benchmarks.loaders.dlmc import DLMCLoader
from spmm_benchmarks.loaders.load import load_dense, load_csr, load_coo

import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
dlmc_loader = DLMCLoader(file_list=SCRIPT_DIR +
                                   "/../../tools/filelists/rn50_magnitude_70_80.txt",
                         loader=load_dense)


printing = True
padding_enabled = True
psc_enabled = False
sop_height = 4
part_per_panel = 4
# class codelet:
#
#     def __init__(self, realpart, imagpart):
#         self.lb = realpart
#         self.i = imagpart


def find_blas(di0f, di1f, di0g, di1g, di0h, di1h):
    ss = 1
    first_i0_ffopd = di0f[0][0]
    first_i1_ffopd = di1f[0][0]
    first_i0_gfopd = di0g[0][0]
    first_i1_gfopd = di1g[0][0]
    first_i0_hfopd = di0h[0][0]
    first_i1_hfopd = di1h[0][0]
    n = len(di0f)
    for i0 in range(0, n, 1):
        m = len(di0f[i0])
        for i1 in range(0,m,1):
            if first_i0_ffopd != di0f[i0][i1] or first_i1_ffopd != di0f[i0][i1]:
                ub_fi0 = i0
                ub_fi1 = m
            if first_i0_gfopd != di0g[i0][i1] or first_i1_gfopd != di0g[i0][i1]:
                ub_gi0 = i0
                ub_gi1 = m
            if first_i0_hfopd != di0h[i0][i1] or first_i1_hfopd != di0h[i0][i1]:
                ub_hi0 = i0
                ub_hi1 = m


# /Users/kazem/UFDB/mesh1e1/mesh1e1.mtx
# /Users/kazem/UFDB/ex5/ex5.mtx
# /Users/kazem/UFDB/LFAT5/LFAT5.mtx
# /Users/kazem/UFDB/bottleneck_3_block_group1_2_1.mtx
def main(argv):
    matrix_path = argv[0]
    mat_name = os.path.basename(matrix_path).split(".")[0]
    out_dir = "spmm_benchmarks/ilp_pad/output"
    A = scipy.io.mmread(matrix_path)
    groups_idx = []
    for matrix, path in dlmc_loader:
        mat_name = os.path.basename(path).split(".")[0]
        matrix = np.array(matrix[0:20, 0:20])
        A_nnz = int(matrix.sum().item())
        A = scipy.sparse.coo_matrix(matrix)
        TI = get_row_col(matrix, A_nnz)
        [Im, Jm, Vm] = TI[1], TI[0], TI[2] #scipy.sparse.find(A)
        plot_matrix(A.nnz, Im, Jm, [], os.path.join(out_dir, mat_name+".png"))
        A.tocsr()
        op_i0, op_i1, op_col = [], [], []
        if psc_enabled:
            [f, g, h, op_i0, op_i1, op_col] = create_functions_spmv_csr(A)
            di0f, di1f = compute_FOPD(f, 2)
            di0g, di1g = compute_FOPD(g, 2)
            di0h, di1h = compute_FOPD(h, 2)
            I, J, V = build_strided_graph(A.shape[1], f, g, h, 1, os.path.join(out_dir, mat_name+'sg.png'))
            sol, dim = psc_mining(I, J, V)
            # sol, dim = TSP(I, J, V)
            groups = list_to_groups(dim, sol)
        else:
            groups_idx, groups_2d, groups, groups_col = sop_mining(matrix, sop_height)
            plot_matrix(A.nnz, Im, Jm, groups, os.path.join(out_dir, mat_name+'sop.png'))

        if padding_enabled:
            num_parts = int(matrix.shape[0] / sop_height) * part_per_panel
            if psc_enabled:
                groups_padd = merging(1, groups, groups_idx, (op_i0, op_col),
                                      psc_enabled, METHOD.ILP)
            else:
                groups_padd = merging_sop(num_parts, groups, groups_idx, (op_i0,
                                                                  op_col), METHOD.ILP)

            print(groups_padd)
            plot_matrix(A.nnz, Im, Jm, groups_padd, os.path.join(out_dir,
                                                             mat_name+'psc_padd.png'))
        print(groups)
        plot_matrix(A.nnz, Im, Jm, groups, os.path.join(out_dir, mat_name+'psc.png'))


if __name__ == "__main__":
    main(sys.argv[1:])