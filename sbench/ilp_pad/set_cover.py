import mosek
import numpy as np
import scipy.sparse
import os
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set, FrozenSet, Iterable, Optional, Union, Any

from utils import *
from mosek_test import *
from functools import partial


class RElement:
    def __init__(self, q_elem, row_idx):
        self.row_idx = row_idx
        self.q_elem = q_elem

    def __repr__(self):
        return f"RElement({self.q_elem}, {self.row_idx})"


QElement = int


@dataclass
class CostModel:
    nCheck: int
    nBase: int
    nCnz: int

    fQ: callable
    fR: callable


def R_from_Q(Q: List[QElement]):
    def bit_locs(x):
        locs = []
        loc = 0
        while x > 0:
            if x & 1:
                locs.append(loc)
            x >>= 1
            loc += 1
        return locs

    print("R",  [RElement(q, row) for q in Q for row in bit_locs(q)])
    return [RElement(q, row) for q in Q for row in bit_locs(q)]


def subset_cost_R(s: List[RElement], cost_model: CostModel):
    # Make sure the index list is all unique
    unique_rows = set([frozenset(elem.row_idx) for elem in s])
    unique_q = set([frozenset(elem.q_elem) for elem in s])

    cost = sum([cost_model.fQ(q) for q in unique_q]) * (cost_model.nBase + unique_rows * cost_model.nCnz)
    cost += cost_model.nCheck

    return cost


def subset_cost_Q(s: List[QElement], cost_model: CostModel, isomorphic: bool):
    # Make sure the index list is all unique
    unique_rows = set([frozenset(elem.row_idx) for elem in s])
    unique_q = set([frozenset(elem.q_elem) for elem in s])

    cost = sum([cost_model.fQ(q) for q in unique_q]) * (cost_model.nBase + unique_rows * cost_model.nCnz)
    cost += cost_model.nCheck

    return cost


# def initial_powerset_codelet_creation(gr_list, height, n_op, occurence_rate_func):
#     p_set = list(powerset(range(1, height+1)))
#     n = len(p_set)
#     real_power_set = []
#     cost, set_to_op = np.zeros((1, n)), np.zeros((n, n_op))
#     for idx_p, ps in enumerate(p_set):
#         gr_ps = Group()
#         for ele in ps:
#             cur_gr = gr_list[ele-1]
#             for nz in cur_gr.nz_lst:
#                 set_to_op[idx_p, nz] = 1
#             gr_ps.absorb_group(cur_gr)
#         cost[0, idx_p] = subset_cost(gr_ps, occurence_rate_func)
#         real_power_set.append(gr_ps)
#     return cost, set_to_op, real_power_set


# def initial_spproximate_superset_creation(A, lst_panel):
#     all_groups, p_groups = [], []
#     for idx, p in enumerate(lst_panel):
#         p_groups.append(initial_codelet_creation(A, p))
#         # all_gr = group_consecutive(p_groups[idx])
#         # print(" group length ", len(p_groups[idx]), " -> ", len(all_gr))
#         # for g in all_gr:
#         #     p_groups[idx].append(g)
#     for i in range(len(p_groups)):
#         for g0 in p_groups[i]:
#             all_groups.append(g0)
#             for j in range(i+1, len(p_groups)):
#                 for g1 in p_groups[j]:
#                     grp = Group()
#                     grp.absorb_group(g0)
#                     grp.absorb_group(g1)
#                     all_groups.append(grp)
#     if len(lst_panel) == 1:
#         all_groups = p_groups[0]
#     print("Len approx power set: ", len(all_groups))
#     return all_groups


# def power_set_creation_approx_new(gr_list, base, n_op, occurence_rate_func):
#     if len(gr_list) <= 16:
#         height = len(gr_list)
#         p_set = list(powerset(range(1, height + 1)))
#     else:
#         p_set = approximate_power_set(len(gr_list), int(base))
#     print(" P SET: ", len(p_set))
#     s2op_I, s2op_J, s2op_V = [], [], []
#     n = len(p_set)
#     real_power_set = []
#     cost = np.zeros((1, n))
#     for idx_p, ps in enumerate(p_set):
#         gr_ps = Group()
#         for ele in ps:
#             cur_gr = gr_list[ele-1]
#             for nz in cur_gr.base_nz:
#                 s2op_I.append(int(idx_p))
#                 s2op_J.append(int(nz))
#                 s2op_V.append(1)
#             gr_ps.absorb_group(cur_gr)
#         cost[0, idx_p] = subset_cost(gr_ps, occurence_rate_func)
#         real_power_set.append(gr_ps)
#     set_to_op = Matrix.sparse(n, n_op, s2op_I, s2op_J, s2op_V)
#     return cost, set_to_op, real_power_set


def cost_constraint_per_codelet(gr_list, n_op, occurence_rate_func):
    n = len(gr_list)
    cost = np.zeros((1, n))
    s2op_I, s2op_J, s2op_V = [], [], []
    for idx_p, ps in enumerate(gr_list):
        cost[0, idx_p] = subset_cost(ps, occurence_rate_func)
        for nz in ps.base_nz:
            s2op_I.append(int(idx_p))
            s2op_J.append(int(nz))
            s2op_V.append(1)
    set_to_op = Matrix.sparse(n, n_op, s2op_I, s2op_J, s2op_V)
    return cost, set_to_op


def set_cover(cost, set_to_op, num_parts):
    n = cost.shape[1]  # 2^16
    #sparse_s2op = scipy.sparse.coo_matrix(set_to_op)
    with Model() as M:
        M.setLogHandler(sys.stdout)
        x = M.variable([1, n], Domain.binary())
        M.constraint(Expr.sum(x, 1), Domain.equalsTo(num_parts))
        #M.constraint(Expr.sum(x, 1), Domain.lessThan(num_parts))
        #M.constraint(Expr.sum(x, 0), Domain.equalsTo(1.0))
        M.constraint(Expr.mul(x, set_to_op), Domain.equalsTo(1.0))
        #M.constraint(x.diag(), Domain.equalsTo(0.))

        M.objective(ObjectiveSense.Minimize, Expr.dot(cost, x))
        M.solve()
        # print(x.level())
        if M.getProblemStatus() == ProblemStatus.Unknown or M.getPrimalSolutionStatus() == SolutionStatus.Unknown:
            return [], n, -1, M.getProblemStatus()
        return x.level(), n, M.primalObjValue(), M.getProblemStatus()


    #tt = approximate_power_set(256, 3)
    #print(tt)

def mining_ncc_cover_set(A, panel_sizes, smart_init, add_super_set, num_parts, occurence_rate_func):
    nnz, final_group = int(A.sum().item()), []
    groups = initial_codelet_creation(A, panel_sizes)
    base = np.min( (math.ceil(np.log2(len(groups)))/2, 4) )
    cost, set_to_op, groups = power_set_creation_approx_new(groups, 3, nnz, occurence_rate_func)
    mrgs_sol, dim, obj, stat = set_cover(cost, set_to_op, num_parts)

    for i in range(mrgs_sol.shape[0]):
        if mrgs_sol[i] > 0.9:
            final_group.append(groups[i])
    return final_group


def set_partitioning(cost, set_to_op):
    n = cost.shape[0]
    with Model() as M:
        M.setLogHandler(sys.stdout)
        x = M.variable([n, n], Domain.binary())
        #M.constraint(Expr.sum(x, 1), Domain.equalsTo(1.0))
        #M.constraint(Expr.sum(x, 0), Domain.equalsTo(1.0))
        M.constraint(Expr.sum(Expr.mul(x, set_to_op), 0), Domain.equalsTo(1.0))
        M.constraint(x.diag(), Domain.equalsTo(0.))
        M.objective(ObjectiveSense.Minimize, Expr.dot(cost, x))
        M.solve()
        # print(x.level())
        if M.getProblemStatus() == ProblemStatus.Unknown or M.getPrimalSolutionStatus() == SolutionStatus.Unknown:
            return [], n, -1, M.getProblemStatus()
        return x.level(), n, M.primalObjValue(), M.getProblemStatus()


def mining_ncc(num_parts, A, panel_sizes):
    groups = initial_overlapping_codelet_creation(A, panel_sizes)
    cur_groups, prev_obj, max_iter, iter_no = len(groups), 1e20, 50, 0
    while cur_groups > num_parts and iter_no < max_iter:
        costs = compute_cost(groups)
        g_to_op = group_to_operation(groups, int(A.sum().item()))
        mrgs_sol, dim, obj, stat = set_partitioning(costs, g_to_op)
        if len(mrgs_sol) == 0:
            print(" Could not reach optimal, just something")
            break
        print("----------->", obj)
        mrgs_schedule = list_to_groups(dim, mrgs_sol)
        groups = create_merged_ncc_list(mrgs_schedule, groups)
        cur_groups = len(groups)
        iter_no += 1
        # if obj >= prev_obj:
        #     break
        # else:
        #     prev_obj = obj
    return groups


def random_matrix_cover_set(argv):
    out_dir = "spmm_benchmarks/ilp_pad/output"
    dir_name = "data2/"
    list_of_paths = os.listdir(dir_name)
    printing, plotting, exporting = True, True, True
    groups_idx = []
    panel_sizes = 4
    num_parts = 50
    for path in list_of_paths:
        print(" ===== Matrix Name ===== ", path)
        matrix, dic_mat, sop_height, wdt = read_unit_codelet_probability(
            os.path.join(dir_name, path))
        mat_name = os.path.basename(path).split(".")[0]
        A_nnz = int(matrix.sum().item())
        A = scipy.sparse.coo_matrix(matrix)
        TI = get_row_col(matrix, A_nnz)
        [Im, Jm, Vm] = TI[1], TI[0], TI[2]  # scipy.sparse.find(A)
        # plot_matrix(A.nnz, Im, Jm, [], os.path.join(out_dir, mat_name+".png"))
        A.tocsr()
        for num_parts in [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4]:
            all_grs = mining_ncc_cover_set(matrix, [8], True, False, num_parts)
            if plotting:
                plot_both(A.nnz, Im, Jm, group_lst_to_nnz_lst(all_grs), os.path.join(
                    out_dir, mat_name + "_" + str(num_parts) +'sop.png'))
            if exporting:
                codelet_list, padding_schedule = convert_group_list_to_dic(all_grs)
                file_out_name = os.path.join(out_dir, mat_name + "_" + str(
                    num_parts)+'.txt')
                list_to_file_dic(codelet_list, file_out_name)
                import_list, import_dic = file_dic_todic(file_out_name)
                if import_dic != padding_schedule:
                    print(" ======== ERROR =========")


def build_basis(m_r):
    edges = set()
    nodes = set()

    def create_label(x):
        return f'0b{x:08b}'

    def ISHIFTC(n, d, N):
        return ((n << d) % (1 << N)) | (n >> (N - d))

    def _build_subgraph_next_level(level, curr_shift):
        level_id = level - 1

        next_level_id = level_id + 1
        next_level = level + 1

        nempty_next = m_r - next_level
        overlap_range = list(range(max(level - nempty_next, 0), level + 1))

        shift = sum([i*x for i, x in enumerate(curr_shift)])
        prev_label = create_label(ISHIFTC((1 << level) - 1, shift % m_r, m_r))


        for i in overlap_range:
            new_shift = curr_shift + [level - i]

            shift = sum([i*x for i, x in enumerate(new_shift)])
            new_label = create_label(ISHIFTC((1 << next_level) - 1, shift % m_r, m_r))

            nodes.add(new_label)

            edge_id = prev_label + new_label
            if next_level > 1 and edge_id not in edges:
                edges.add(edge_id)

            if level < (m_r - 1):
                _build_subgraph_next_level(level + 1, new_shift)


    curr_shift = [0]
    label = create_label(1)
    nodes.add(label)

    _build_subgraph_next_level(1, curr_shift)

    return sorted(list(nodes))


def random_matrix_cover_set_2(argv):
    out_dir = "spmm_benchmarks/ilp_pad/output"
    dir_name = "data4/"

    printing, plotting, exporting = True, True, True
    groups_idx = []
    panel_size = 4
    num_parts = 10

    SORT_BY_NNZ_COUNT = True

    def build_synthetic_matrix_from_patterns(rows, patterns):
        A = np.zeros((rows, len(patterns)))
        for col, pattern in enumerate(patterns):
            for row in range(rows):
                if (1 << row) & pattern:
                    A[row][col] = 1
        return A

    for sp in [0.7, 0.8, 0.9, 0.95]:
        mat_name = f'random_{int(sp*100)}'

        print(f' ===== {int(sp*100)}% Sparse (Random) ===== ', sp)
        Q_str = build_basis(panel_size)
        Q = [int(x, 2) for x in Q_str]

        # Sort by increasing NNZ count for the sake of visualization
        if SORT_BY_NNZ_COUNT:
            import gmpy; Q = sorted(Q, key=lambda x: gmpy.popcount(x))

        R = R_from_Q(Q)

        print(Q)
        print(R)
        print(len(Q), len(R))

        matrix = build_synthetic_matrix_from_patterns(panel_size, possible_patterns)

        def rand_matrix_occurence_rate(idx_list, sparsity):
            # TODO: figure out how to prevent double counting due to overlap
            density = (1-sparsity)
            return density**len(idx_list)

        occurence_rate_func = partial(rand_matrix_occurence_rate, sparsity=sp)

        A_nnz = int(matrix.sum().item())
        A = scipy.sparse.coo_matrix(matrix)
        TI = get_row_col(matrix, A_nnz)
        [Im, Jm, Vm] = TI[1], TI[0], TI[2] #scipy.sparse.find(A)

        A.tocsr()
        for num_parts in range(40, 140, 10):
            all_grs = mining_ncc_cover_set(matrix, panel_size, True,
                                           True, num_parts, occurence_rate_func)
            if plotting:
                plot_both(A.nnz, Im, Jm, group_lst_to_nnz_lst(all_grs), os.path.join(
                    out_dir, mat_name+ "_"+str(num_parts)+'sop.png'))
            if exporting:
                codelet_list, padding_schedule = convert_group_list_to_dic(all_grs)
                file_out_name = os.path.join(out_dir, mat_name + "_" + str(
                    num_parts)+'.txt')
                list_to_file_dic(codelet_list, file_out_name)
                import_list, import_dic = file_dic_todic(file_out_name)
                if import_dic != padding_schedule:
                    print(" ======== ERROR =========")



def random_matrix_test(argv):
    out_dir = "spmm_benchmarks/ilp_pad/output"
    dir_name = "data4/"
    list_of_paths = os.listdir(dir_name)
    printing, plotting = True, True
    groups_idx = []
    panel_sizes = 4
    num_parts = 10
    for path in list_of_paths:
        print(" ===== Matrix Name ===== ", path)
        matrix, dic_mat, sop_height, wdt = read_unit_codelet_probability(
            os.path.join(dir_name, path))
        mat_name = os.path.basename(path).split(".")[0]
        A_nnz = int(matrix.sum().item())
        A = scipy.sparse.coo_matrix(matrix)
        TI = get_row_col(matrix, A_nnz)
        [Im, Jm, Vm] = TI[1], TI[0], TI[2] #scipy.sparse.find(A)
        #plot_matrix(A.nnz, Im, Jm, [], os.path.join(out_dir, mat_name+".png"))
        A.tocsr()
        part_array = []
        if sop_height == 4:
            part_array = range(2, 15, 1)
        else:
            part_array = range(8, 80, 2)
        #num_parts = 40 if sop_height == 8 else 8
        for num_parts in part_array:
            all_grs = mining_ncc(num_parts, matrix, [sop_height])
            if plotting:
                plot_both(A.nnz, Im, Jm, group_lst_to_nnz_lst(all_grs), os.path.join(
                    out_dir, mat_name+'sop.png'))
            codelet_list, padding_schedule = convert_group_list_to_dic(all_grs)
            file_out_name = os.path.join(out_dir, mat_name + "_" + str(
                num_parts)+'.txt')
            list_to_file_dic(codelet_list, file_out_name)
            import_list, import_dic = file_dic_todic(file_out_name)
            if import_dic != padding_schedule:
                print(" ======== ERROR =========")


if __name__ == "__main__":

    random_matrix_cover_set_2(sys.argv[1:])
    #random_matrix_test(sys.argv[1:])
