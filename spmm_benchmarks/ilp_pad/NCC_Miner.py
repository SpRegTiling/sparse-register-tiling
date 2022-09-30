import mosek
import numpy as np
import scipy.sparse
import os
import math


from utils import *
from mosek_test import *
from functools import partial


class Group:
    def __init__(self):
        self.base_row_idx = []
        self.base_col_idx = []
        self.base_nz = []
        self.idx_list = []
        self.col_idx_lst = []
        self.nz_lst = []

    def add_group_row_idx(self, new_idx):
        self.base_row_idx = list(set(self.base_row_idx + new_idx))
        self.idx_list.append(new_idx)

    def add_group_col_idx(self, new_col_idx):
        self.base_col_idx = list(set(self.base_col_idx + new_col_idx))
        self.col_idx_lst.append(new_col_idx)

    def add_group_nz(self, new_nz):
        self.base_nz = list(set(self.base_nz + new_nz))
        self.nz_lst.append(new_nz)

    def get_num_op(self):
        all_ops, real_ops = 0, 0
        all_ops = len(self.base_row_idx) * len(self.base_col_idx)
        real_ops = len(self.base_nz)  # assumes nz is accessed once
        return all_ops, real_ops

    def get_nnz_group(self):
        return self.base_nz

    def absorb_group(self, in_gr):
        for f in in_gr.idx_list:
            self.add_group_row_idx(f)
        for f in in_gr.col_idx_lst:
            self.add_group_col_idx(f)
        for f in in_gr.nz_lst:
            self.add_group_nz(f)


def is_equal(grp1, grp2):
    if grp1.base_row_idx == grp2.base_row_idx:
        return True
    if grp1.base_col_idx == grp2.base_col_idx:
        return True
    return False


def convert_group_list_to_dic(grp_list):
    ### convert the mined list to dictionary
    bucket_pattern, pattern_dic = [], {}
    for idx_g, gr in enumerate(grp_list):
        value_dic_bin, value_dic = idx_list_to_binary(gr.base_row_idx, 16)
        bucket_pattern.append([value_dic])
        for row_idx in gr.idx_list:
            key_dic_bin, key_dic = idx_list_to_binary(row_idx, 16)
            pattern_dic[key_dic] = value_dic
            bucket_pattern[idx_g].append(key_dic)
    return bucket_pattern, pattern_dic


def group_consecutive(grp_list):
    agg_list, idx = [], 0
    while idx < len(grp_list)-1:
        grp1, j = grp_list[idx], idx+1
        while is_equal(grp1, grp_list[j]):
            grp1.absorb_group(grp_list[j])
            j = j+1
            if j >= len(grp_list):
                break
        agg_list.append(grp1)
        if j == len(grp_list)-1 and not is_equal(grp_list[j-1], grp_list[j]):
            agg_list.append(grp_list[j])
        idx = j
    return agg_list


def merge_ncc(ncc1, ncc2):
    merged_ncc = Group()
    merged_ncc.absorb_group(ncc1)
    merged_ncc.absorb_group(ncc2)
    return merged_ncc


def pair_NCC_cost(ncc1, ncc2):
    blas_grp = []
    tot_ops1, real_ops1 = ncc1.get_num_op()
    tot_ops2, real_ops2 = ncc2.get_num_op()
    t_ops_merged, r_ops_merged = merge_ncc(ncc1, ncc2).get_num_op()
    cost_grp = t_ops_merged / (real_ops1+real_ops2) #+ (100/r_ops_merged)
    return blas_grp, cost_grp


def compute_cost(codelet_groups):
    grp_no = len(codelet_groups)
    grp_mat = np.zeros((grp_no, grp_no))
    for i in range(grp_no):
        for j in range(i, grp_no):
            [gr, cst] = pair_NCC_cost(codelet_groups[i], codelet_groups[j])
            grp_mat[i, j] = cst
            grp_mat[j, i] = cst
    return grp_mat


def initial_codelet_creation(A, row_panel_size):
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
    all_groups = []
    num_panels = int(n/row_panel_size)
    for i in range(0, num_panels):
        groups_2d.append([])
        mat = A[i*row_panel_size:(i+1)*row_panel_size, :]
        for j in range(A.shape[1]):
            if np.sum(np.ravel(mat[:, j])) != 0:
                tmp = take_nonzeros(T[i*row_panel_size:(i+1)*row_panel_size,j, 2], T[i*row_panel_size:(i+1)*row_panel_size,j, 0])
                tmp_col = take_nonzeros(T[i*row_panel_size:(i+1)*row_panel_size,j, 2], T[i*row_panel_size:(i+1)*row_panel_size,j, 1])
                tmp_nz = take_nonzeros(T[i*row_panel_size:(i+1)*row_panel_size,j, 2], T[i*row_panel_size:(i+1)*row_panel_size,j, 3])
                gr = Group()
                gr.add_group_nz(tmp_nz)
                gr.add_group_row_idx(tmp)
                gr.add_group_col_idx(tmp_col)
                all_groups.append(gr)
    return all_groups


def initial_overlapping_codelet_creation(A, lst_panel):
    all_groups = []
    for p in lst_panel:
        gr = initial_codelet_creation(A, p)
        for g in gr:
            all_groups.append(g)
    all_groups = group_consecutive(all_groups)
    return all_groups


def group_to_operation(groups, tot_ops):
    group_to_op = np.zeros((len(groups), tot_ops))
    for idx, g in enumerate(groups):
        for nz in g.base_nz:
            group_to_op[idx, nz] = 1
    return group_to_op


def group_lst_to_nnz_lst(grp_lst):
    nz_lst = []
    for g in grp_lst:
        nz_lst.append(g.get_nnz_group())
    return nz_lst


def create_merged_ncc_list(merg_schedule, groups):
    new_grp = []
    for m in merg_schedule:
        merged_set = Group()
        for i in m:
            merged_set.absorb_group(groups[i])
        new_grp.append(merged_set)
    return new_grp


def subset_cost(group: Group, occurence_rate_func):
    NNZ_COST = 1        # Represents the variable cost of an SOP, i.e. cost of loading matrix A + FMA
    BASE_COST = 2       # Represents the baseline cost of an SOP, i.e. cost of loading matrix B

    # Make sure the index list is all unique
    unique_idx_lists = set([frozenset(idx_list) for idx_list in group.idx_list])
    num_unique_idx_lists = len(unique_idx_lists)

    # This assert triggers when using overlapping panels, so instead we'll just dedupe it here
    #   this can happen if you have a [1, 0, 0, 0] SOP4 and [1, 0] SOP2
    #assert num_unique_idx_lists == len(group.idx_list)

    unique_nnz_locations = len(set([loc for idx_list in group.idx_list for loc in idx_list]))
    group_occurence_rate = sum([occurence_rate_func(idx_list) for idx_list in unique_idx_lists])

    # group_occurence_rate acts as the width of the merged group
    total_ops = unique_nnz_locations * group_occurence_rate * NNZ_COST + group_occurence_rate * BASE_COST

    return total_ops


def initial_powerset_codelet_creation(gr_list, height, n_op, occurence_rate_func):
    p_set = list(powerset(range(1, height+1)))
    n = len(p_set)
    real_power_set = []
    cost, set_to_op = np.zeros((1, n)), np.zeros((n, n_op))
    for idx_p, ps in enumerate(p_set):
        gr_ps = Group()
        for ele in ps:
            cur_gr = gr_list[ele-1]
            for nz in cur_gr.nz_lst:
                set_to_op[idx_p, nz] = 1
            gr_ps.absorb_group(cur_gr)
        cost[0, idx_p] = subset_cost(gr_ps, occurence_rate_func)
        real_power_set.append(gr_ps)
    return cost, set_to_op, real_power_set


def initial_spproximate_superset_creation(A, lst_panel):
    all_groups, p_groups = [], []
    for idx, p in enumerate(lst_panel):
        p_groups.append(initial_codelet_creation(A, p))
        # all_gr = group_consecutive(p_groups[idx])
        # print(" group length ", len(p_groups[idx]), " -> ", len(all_gr))
        # for g in all_gr:
        #     p_groups[idx].append(g)
    for i in range(len(p_groups)):
        for g0 in p_groups[i]:
            all_groups.append(g0)
            for j in range(i+1, len(p_groups)):
                for g1 in p_groups[j]:
                    grp = Group()
                    grp.absorb_group(g0)
                    grp.absorb_group(g1)
                    all_groups.append(grp)
    if len(lst_panel) == 1:
        all_groups = p_groups[0]
    print("Len approx power set: ", len(all_groups))
    return all_groups


def power_set_creation_approx_new(gr_list, base, n_op, occurence_rate_func):
    if len(gr_list) <= 16:
        height = len(gr_list)
        p_set = list(powerset(range(1, height + 1)))
    else:
        p_set = approximate_power_set(len(gr_list), int(base))
    print(" P SET: ", len(p_set))
    s2op_I, s2op_J, s2op_V = [], [], []
    n = len(p_set)
    real_power_set = []
    cost = np.zeros((1, n))
    for idx_p, ps in enumerate(p_set):
        gr_ps = Group()
        for ele in ps:
            cur_gr = gr_list[ele-1]
            for nz in cur_gr.base_nz:
                s2op_I.append(int(idx_p))
                s2op_J.append(int(nz))
                s2op_V.append(1)
            gr_ps.absorb_group(cur_gr)
        cost[0, idx_p] = subset_cost(gr_ps, occurence_rate_func)
        real_power_set.append(gr_ps)
    set_to_op = Matrix.sparse(n, n_op, s2op_I, s2op_J, s2op_V)
    return cost, set_to_op, real_power_set


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
        possible_patterns = list(range(1, 2**panel_size+1))
        Q = build_basis(panel_size)
        Q = [int(x, 2) for x in Q]

        # Sort by increasing NNZ count for the sake of visualization
        if SORT_BY_NNZ_COUNT:
            import gmpy; Q = sorted(Q, key=lambda x: gmpy.popcount(x))

        print(Q)

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
