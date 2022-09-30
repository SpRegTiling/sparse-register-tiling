import numpy as np
from mosek_test import *
from utils import *



def flatten(xss):
    return [x for xs in xss for x in xs]


def sop_mining(A, row_panel_size):
    n = A.shape[0]
    cnt = 0
    T = np.zeros((n, A.shape[1], 4))
    for i in range(n):
        for j in range(A.shape[1]):
            T[i, j, 0] = i
            T[i, j, 1] = j
            T[i, j, 2] = A[i, j]
            if A[i, j] != 0:
                T[i, j, 3] = cnt
                cnt = cnt + 1
    groups, groups_2d, group_nz, groups_cols, groups_val = [], [], [], [], []
    num_panels = int(n/row_panel_size)
    for i in range(0, num_panels):
        groups_2d.append([])
        mat = A[i*row_panel_size:(i+1)*row_panel_size, :]
        for j in range(A.shape[1]):
            if np.sum(np.ravel(mat[:, j])) != 0:
                tmp = take_nonzeros(T[i*row_panel_size:(i+1)*row_panel_size,j, 2], T[i*row_panel_size:(i+1)*row_panel_size,j, 0])
                tmp_col = take_nonzeros(T[i*row_panel_size:(i+1)*row_panel_size,j, 2], T[i*row_panel_size:(i+1)*row_panel_size,j, 1])
                tmp_nz = take_nonzeros(T[i*row_panel_size:(i+1)*row_panel_size,j, 2], T[i*row_panel_size:(i+1)*row_panel_size,j, 3])
                groups.append(tmp)
                groups_cols.append(tmp_col)
                groups_2d[i].append(tmp)
                group_nz.append(tmp_nz)

    return groups, groups_2d, group_nz, (np.array(flatten(groups_cols)),
                                         np.array(flatten(groups)))


def padding_blas_cost(pnt_grp1, pnt_grp2, op_to_idx):
    blas_grp = []
    min_x, min_y = 1e20, 1e20
    max_x, max_y = 0, 0
    merged_grp = list(set(pnt_grp1 + pnt_grp2))
    for pnt in merged_grp:
        x_coo = op_to_idx[0][pnt]
        y_coo = op_to_idx[1][pnt]
        min_x = np.min((x_coo, min_x))
        min_y = np.min((y_coo, min_y))
        max_x = np.max((x_coo, max_x))
        max_y = np.max((y_coo, max_y))
    cost_grp = (max_x-min_x+1) * (max_y-min_y+1) - len(merged_grp)
    return blas_grp, cost_grp


def padding_sop_cost(pnt_grp1, pnt_grp2, op_to_idx):
    blas_grp = []
    l1 = len(pnt_grp1)
    l2 = len(pnt_grp2)
    merged_grp_l = len(list(set(pnt_grp1 + pnt_grp2)))
    cost_grp = merged_grp_l / (l1+l2)
    return blas_grp, cost_grp


def pair_group_cost(grp1, grp2):
    blas_grp = []
    l1 = grp1.get_num_op()
    l2 = len(grp2)
    merged_grp_l = len(list(set(grp1 + grp2)))
    cost_grp = merged_grp_l / (l1+l2)
    return blas_grp, cost_grp


def BLAS_padding(codelet_groups, operation_to_coordinate, psc_enabled):
    grp_no = len(codelet_groups)
    grp_mat = np.zeros((grp_no, grp_no))
    for i in range(grp_no):
        for j in range(i, grp_no):
            if psc_enabled:
                [gr, cst] = padding_blas_cost(codelet_groups[i], codelet_groups[j],
                                              operation_to_coordinate)
            else:
                [gr, cst] = padding_sop_cost(codelet_groups[i], codelet_groups[j],
                                             operation_to_coordinate)
            grp_mat[i, j] = cst
            grp_mat[j, i] = cst
    return grp_mat


def create_merged_group(merg_schedule, groups):
    new_grp = []
    for m in merg_schedule:
        merged_set = []
        for i in m:
            merged_set = list(set(merged_set + groups[i]))
        new_grp.append(merged_set)
    return new_grp


def merging(num_iter, init_grps, init_grps_idx, op_to_coo, psc_enabled, method):
    groups = init_grps if psc_enabled else init_grps_idx
    for i in range(num_iter):
        costs = BLAS_padding(groups, op_to_coo, psc_enabled)
        if method == METHOD.ILP:
            mrgs_sol, dim = padding_ilp_problem(costs)
        else:
            mrgs_sol, dim = padding_sorting_problem(costs)
        mrgs_schedule = list_to_groups(dim, mrgs_sol)
        groups = create_merged_group(mrgs_schedule, groups)
        groups = create_merged_group(mrgs_schedule, init_grps)
    return groups


def merging_sop(num_parts, init_grps, init_grps_idx, op_to_coo, method):
    groups = init_grps_idx
    groups_nnz = init_grps
    cur_groups = len(groups)
    while cur_groups > num_parts:
        costs = BLAS_padding(groups, op_to_coo, False)
        if method == METHOD.ILP:
            mrgs_sol, dim = padding_ilp_problem(costs)
        else:
            mrgs_sol, dim = padding_sorting_problem(costs)
        mrgs_schedule = list_to_groups(dim, mrgs_sol)
        groups = create_merged_group(mrgs_schedule, groups)
        groups_nnz = create_merged_group(mrgs_schedule, groups_nnz)
        cur_groups = len(groups)
    return groups_nnz








