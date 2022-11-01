
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import graphviz
import matplotlib.cm as cm
from random import randint
import random
from enum import Enum

from itertools import chain, combinations

class METHOD(Enum):
    ILP = 1
    SORT = 2

# Utility function to create dictionary
def multi_dict(K, type):
    if K == 1:
        return defaultdict(type)
    else:
        return defaultdict(lambda: multi_dict(K - 1, type))


# converts 2D list to triplet form
def list_to_triplet(l, c):
    n = len(l)
    I = []
    J = []
    V = []
    for i in range(n):
        for j in range(len(l[i])):
            # if i > l[i][j]:
            #     continue
            I.append(i)
            J.append(l[i][j])
            V.append(c[i][j])
    return I, J, V


def plot_graph(g, cost, name):
    ps = graphviz.Graph(name, node_attr={'shape': 'plaintext'}, format='png', engine='fdp')
    n = len(g)
    for i in range(n):
        v1 = str(i)
        ps.node(v1, shape='circle')
        for j in range(len(g[i])):
            if i >= g[i][j]:
                continue
            v2 = str(g[i][j])
            ps.node(v2)
            ps.edge(v1, v2, label=str(cost[i][j]))
            #ps.edge(v1, v2)
    ps.render(view=False)


def list_to_groups(n, sol):
    groups = []
    visited = np.full((n), False)
    for i in range(n):
        if visited[i]:
            continue
        tmp_grp = []
        tmp_grp.append(i)
        visited[i] = True
        tmp = []
        c_node = np.where(sol[i*n:(i+1)*n] == 1)[0]
        for k in c_node:
            if not visited[k]:
                tmp.append(k)
        while len(tmp) != 0:
            j = tmp[0]
            tmp = tmp[1:]
            tmp_grp.append(j)
            visited[j] = True
            c_node = np.where(sol[j * n:(j + 1) * n] == 1)[0]
            for k in c_node:
                if not visited[k] and k not in tmp:
                    tmp.append(k)
        groups.append(tmp_grp)
    return groups


def plot_matrix(nnz, col, row, groups, output_path):
    fig, axi = plt.subplots(2, figsize=(10, 6))
    colors = random.sample(range(0, 0xFFFFFF), len(groups))
    for i in range(len(groups)):
        colors[i] = '#%06X' % colors[i]
    x_axis = []
    y_axis = []
    for i in range(len(groups)):
        x_axis.append(col[groups[i][:]])
        y_axis.append(row[groups[i][:]])
    if len(groups) >= 1:
        for x, y, color in zip(x_axis, y_axis, colors):
            axi[0].scatter(x, y, color=color)
            ax = plt.gca()  # get the axis
            #ax.set_xlim(ax.get_xlim()[::-1])  # invert the axis
            ax.xaxis.tick_top()  # and move the X-Axis
            #ax.set_ylim(ax.get_ylim()[::-1])
            #ax.yaxis.set_ticks(np.arange(0, max(col)+1, 2))  # set y-ticks
            ax.yaxis.tick_left()  # remove right y-Ticks
    else:
        for i in range(nnz):
            axi[0].scatter(col[i], row[i], color='b')
        ax = plt.gca()  # get the axis
        ax.set_ylim(ax.get_ylim()[::-1])  # invert the axis
        ax.xaxis.tick_top()  # and move the X-Axis
        #ax.yaxis.set_ticks(np.arange(0, 16, 1))  # set y-ticks
        ax.yaxis.tick_left()  # remove right y-Ticks

    plt.savefig(output_path)
    return plt


def plot_both(nnz, col, row, groups, output_path):
    fig, axi = plt.subplots(2, figsize=(20, 6))
    colors = random.sample(range(0, 0xFFFFFF), len(groups))
    for i in range(len(groups)):
        colors[i] = '#%06X' % colors[i]
    x_axis = []
    y_axis = []
    for i in range(len(groups)):
        x_axis.append(col[groups[i][:]])
        y_axis.append(row[groups[i][:]])
    for x, y, color in zip(x_axis, y_axis, colors):
        axi[0].scatter(x, y, color=color)
        ax = plt.gca()  # get the axis
       # ax.set_ylim(ax.get_ylim()[::-1])  # invert the axis
        ax.xaxis.tick_top()  # and move the X-Axis
        ax.yaxis.tick_left()  # remove right y-Ticks

    for i in range(nnz):
        axi[1].scatter(col[i], row[i], color='b')
    ax = plt.gca()  # get the axis
    ax.xaxis.tick_top()  # and move the X-Axis
    ax.yaxis.tick_left()  # remove right y-Ticks
    plt.savefig(output_path)
    return plt


def sort_matrix(mat):
    coo_list = []
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            coo_list.append((i, j, mat[i, j]))
    #coo_list.sort(key= lambda y: y[2])
    coo_list = sorted(coo_list, key=lambda tup: tup[2])
    #print(coo_list)
    return coo_list


def get_row_col(A, nnz):
    n = A.shape[0]
    cnt = 0
    T = np.zeros((4, nnz))
    for i in range(n):
        for j in range(A.shape[1]):
            if A[i, j] != 0:
                T[0, cnt] = i
                T[1, cnt] = j
                T[2, cnt] = A[i, j]
                cnt = cnt+1
    return T


def idx_list_to_binary(idx_list, length):
    bin_vec = np.zeros(length)
    int_val = 0
    for i in idx_list:
        bin_vec[i] = 1
        int_val += pow(2, i)
    return bin_vec, int_val


def list_to_file_dic(in_list, file_out):
    file1 = open(file_out, 'w')
    for item in in_list:
        for idx_k, keys in enumerate(item):
            if idx_k == len(item)-1:
                file1.write(str(keys))
            else:
                file1.write(str(keys) + ",")
        file1.write("\n")
    file1.close()


def file_dic_todic(file_in):
    dic_pat, value_list = {}, []
    file1 = open(file_in, 'r')
    lines = file1.readlines()
    for line in lines:
        entry = line.split(',')
        cur_value = 0
        for idx_v, v in enumerate(entry):
            if idx_v == 0:
                cur_value = int(v)
                value_list.append(cur_value)
            else:
                dic_pat[int(v)] = cur_value
    return value_list, dic_pat

def pattern_to_vec(pat, code_beg):
    sp_pat = list(pat)
    sp_pat = sp_pat[len(sp_pat)-code_beg:]
    vec = np.zeros(len(sp_pat))
    for idx, p in enumerate(sp_pat):
        if p == '1':
            vec[idx] = 1
    return vec


def read_unit_codelet_probability(file_name):
    uc_to_prob = {}
    uc_to_wdth = {}
    wd, zero_wdth = 0, 0
    file1 = open(file_name, 'r')
    lines = file1.readlines()
    if len(lines) == 16:
        width = 15
    elif len(lines) == 4:
        width = 3
    else:
        width = 256
    for line in lines:
        uc, prob = line.split(',')[0].strip(), float(line.split(',')[1].strip())
        if uc == '0b00000000':
            continue
        uc_to_prob[uc] = prob
        tmp = np.max((np.floor(prob * width), 1))
        uc_to_wdth[uc], wd = tmp, wd+tmp
    if not wd == width:
        topup = np.abs(width - wd)
        flag = ((width-wd) > 0)
        while topup > 0:
            for i in uc_to_wdth.items():
                if flag:
                    uc_to_wdth[i[0]] = uc_to_wdth[i[0]]+1
                    topup = topup - 1
                else:
                    if uc_to_wdth[i[0]] > 1:
                        uc_to_wdth[i[0]] = uc_to_wdth[i[0]]-1
                        topup = topup - 1
                if topup == 0:
                    break
    dim = int(np.log2(len(lines)))
    matrix = np.zeros((dim, width))
    beg, end = 0, 0
    sss = 0
    for i in uc_to_wdth.items():
        sss += i[1]
    for i in uc_to_wdth.items():
        v = pattern_to_vec(i[0], dim)
        end = int(beg + i[1])
        for j in range(beg, end, 1):
            matrix[:, j] = v
        beg = end
    return matrix, uc_to_prob, dim, width



def take_nonzeros(a1, a2):
    nz_idx = []
    for i in range(a1.shape[0]):
        if a1[i] != 0:
            nz_idx.append(int(a2[i]))
    return nz_idx


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def approximate_power_set(set_size, base_size):
    power_set, sw = [], False
    if base_size > 4:
        return power_set
    height = pow(2, base_size)
    b_set = list(powerset(range(1, height+1)))
    num_ss = set_size // height
    for i in range(num_ss+1):
        for s1 in b_set:
            new_se = []
            sw = False
            for s2 in s1:
                if s2 + height*i > set_size:
                    sw = True
                new_se.append(s2 + height * i)
            if len(new_se) > 0 and not sw:
                power_set.append(new_se)
    return power_set
