import numpy as np
import scipy

from utils import *

class codeket_model:
    def __init__(self):
        self.f_a = []
        self.f_b = []
        self.f_init = []
        self.g_a = []
        self.g_b = []
        self.g_init = []
        self.h_a = []
        self.h_b = []
        self.h_init = []
        self.i0 = []
        self.i1 = []


def compute_FOPD(ff, dim):
    di0 = []
    di1 = []
    if dim == 2:
        i0_len = len(ff)
        for i0 in range(i0_len-1):
            i1_len = len(ff[i0])
            i1p1_len = len(ff[i0+1])
            min_both = min(i1_len, i1p1_len)
            tmp = np.zeros(min_both-1 if min_both>0 else 0, dtype=int)
            for i1 in range(min_both-1):
                tmp[i1] = ff[i0 + 1][i1] - ff[i0][i1]
            di0.append(tmp)
            tmp = np.zeros(i1_len-1 if i1_len>0 else 0, dtype=int)
            for i1 in range(i1_len-1):
                tmp[i1] = ff[i0][i1+1] - ff[i0][i1]
            di1.append(tmp)
    return di0, di1


# Takes FOPDs of three functions as input and returns geometric distance cost
def compute_cost(dfdi0, dfdi1, dgdi0, dgdi1, dhdi0, dhdi1):
    data_space_no = 3
    num_unstrided = data_space_no
    if (abs(dfdi0) == 1 or abs(dfdi0) == 0) and (abs(dfdi1) == 1 or abs(dfdi1) == 0):
        num_unstrided -= 1
    if (abs(dgdi0) == 1 or abs(dgdi0) == 0) and (abs(dgdi1) == 1 or abs(dgdi1) == 0):
        num_unstrided -= 1
    if (abs(dhdi0) == 1 or abs(dhdi0) == 0) and (abs(dhdi1) == 1 or abs(dhdi1) == 0):
        num_unstrided -= 1
    if abs(dfdi0) > 1:
        dfdi0 = 2
    if abs(dgdi0) >= 1:
        dgdi0 = 4
    if abs(dhdi0) > 1:
        dhdi0 = 2
    g_dist = abs(dfdi0) + abs(dfdi1) + abs(dgdi0) + abs(dgdi1) + abs(dhdi0) + abs(dhdi1)
    return data_space_no-num_unstrided, data_space_no, g_dist


def build_strided_graph(n, f, g, h, ns_thr, out_path):
    d = 1
    strided_graph = []
    cost = []
    op_cnt = 0;
    # for every nonzero, compute edge weight
    for i0 in range(n):
        for i1 in range(len(f[i0])):
            tmp_op = []
            tmp_cost = []
            op_cnt_tar = 0
            for k0 in range(n):
                for k1 in range(len(f[k0])):
                    if i0 == k0 and i1 == k1:
                        op_cnt_tar+=1
                        continue
                    df = f[i0][i1] - f[k0][k1]
                    dg = g[i0][i1] - g[k0][k1]
                    dh = h[i0][i1] - h[k0][k1]
                    n_strided, n_ds, dist = compute_cost(df, 0, dg, 0, dh, 0)
                    if n_strided >= ns_thr: # add an edge
                        tmp_op.append(op_cnt_tar)
                        #dist=1
                        tmp_cost.append(dist*(n_ds-n_strided)*(n_ds-n_strided))
                    op_cnt_tar += 1
            strided_graph.append(tmp_op)
            cost.append(tmp_cost)
    I, J, V = list_to_triplet(strided_graph, cost)
    #plot_graph(strided_graph, cost, out_path)
    return I, J, V

def create_functions_spmv_csr(A):
    ff = []
    gg = []
    hh = []
    n = A.shape[1]
    [I, J, V] = scipy.sparse.find(A)
    nnz = len(I)
    opno_to_i0, opno_to_i1, opno_to_col = [], [], []
    prev = 0
    inner_iter = 0
    row_len = np.zeros(n, dtype=int)
    for k in range(nnz):
        row_len[J[k]] += 1
    for k in range(n):
        inner_iter = row_len[k]
        ff.append(np.zeros(inner_iter, dtype=int))
        gg.append(np.zeros(inner_iter, dtype=int))
        hh.append(np.zeros(inner_iter, dtype=int))

    cnz = inner_iter = 0
    for k in range(n):
        for l in range(row_len[k]):
            i0 = k
            i1 = l
            ff[i0][i1] = i0
            gg[i0][i1] = cnz
            hh[i0][i1] = I[cnz]
            opno_to_i0.append(i0)
            opno_to_i1.append(i1)
            opno_to_col.append(I[cnz])
            cnz += 1
        #print (k, ": ", i0, ", ", i1, ", ", I[i1], " \n")
    return ff, gg, hh, opno_to_i0, opno_to_i1, opno_to_col




def opno_to_access_func(grp, op2i0, op2i1, f, g, h):
    ff = []
    gg = []
    hh = []
    tmp_f = []
    tmp_g = []
    tmp_h = []
    first_i0 = op2i0[grp[0]]
    for n in grp:
        if op2i0[n] == first_i0:
            tmp_f.append(f[op2i0[n]][op2i1[n]])
            tmp_g.append(g[op2i0[n]][op2i1[n]])
            tmp_h.append(h[op2i0[n]][op2i1[n]])
        else:
            ff.append(tmp_f)
            gg.append(tmp_g)
            hh.append(tmp_h)
            tmp_f = []
            tmp_g = []
            tmp_h = []
            first_i0 = op2i0[grp[n]]
    return ff, gg, hh


def get_bounds(dfdi0, dfdi1):
    cm = codeket_model()
    if dfdi0.all(element == dfdi0[0] for element in dfdi0):
        cm.f_a = dfdi0[0]

    if dfdi1.all(element == dfdi1[0] for element in dfdi1):
        cm.f_b = dfdi1[0]


def mine_for_PSC(f, g, h):
    dfdi0, dfdi1 = compute_FOPD(f, 2)
    dgdi0, dgdi1 = compute_FOPD(g, 2)
    dhdi0, dhdi1 = compute_FOPD(h, 2)
    for i in range(0, len(f), 2):
        if dfdi0[i].all(element == dfdi0[i][0] for element in dfdi0[i]) and dfdi0[i].all(element == dfdi0[i][0] for element in dfdi0[i]):
            print("f")



class codelet:
    def __init__(self, i0, i1, f, g, h):
        # f, g, and h are large functions
        self.i0 = i0
        self.i1 = i1
        self.f = f
        self.g = g
        self.h = h
        self.dfdi0 = []
        self.dfdi1 = []
        self.dgdi0 = []
        self.dgdi1 = []
        self.dhdi0 = []
        self.dhdi1 = []
        self.type = 1
        self.variable_space = 0
        self.num_strided = 0

    def codelet_type(self, f, g, h):
        # Checking all space to fit it into one type.
        dfdi0, dfdi1 = compute_FOPD(f, 2)
        dgdi0, dgdi1 = compute_FOPD(g, 2)
        dhdi0, dhdi1 = compute_FOPD(h, 2)
        # first check the iterations space is equal or not
        for i in range(len(f)):
            if len(f[i]) != len(g[i]) != len(h[i]):
                # check whether dfdi0 is strided
                variable_space = 1  # PSC I
        for i in range(0, len(f), 2):
            if dfdi0[i].all(element == dfdi0[i][0] for element in dfdi0[i]) and dfdi0[i].all(
                    element == dfdi0[i][0] for element in dfdi0[i]):
                print("f")

        for i in self.i0:
            if self.dfdi0[i][:] == self.dfdi0[0][0] or self.dfdi1[i][:] == self.dfdi1[0][0]: # or g or h
                # check whether dfdi0 is strided
                self.num_strided += 1  # PSC I
        # then check FOPDs on f

    def get_combination(self):
        s = 1

