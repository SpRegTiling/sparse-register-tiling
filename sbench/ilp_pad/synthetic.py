import scipy.io
import scipy.sparse
import sys, os

from utils import *
from psc_codelet import *
from mosek_test import psc_mining, TSP
from padding import *

dir_name = "/home/kazem/development/dnn-spmm-bench/data/"
list_of_paths = os.listdir(dir_name)
printing = True
padding_enabled = True
psc_enabled = False
sop_height = 4
part_per_panel = 40


def main(argv):
    matrix_path = ""
    mat_name = ""
    out_dir = "/home/kazem/development/dnn-spmm-bench/spmm_benchmarks/ilp_pad/output"
    groups_idx = []
    #for matrix, path in dlmc_loader:
    for path in list_of_paths:
        matrix, dic_mat, sop_height, wdt = read_unit_codelet_probability(os.path.join(dir_name,path))
        mat_name = os.path.basename(path).split(".")[0]
        #matrix = np.array(matrix[0:20, 0:20])
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
            groups_idx, groups_2d, groups, groups_col, all_grs = sop_mining(matrix, sop_height)
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
