import math
import torch
import sop_driver
import numpy as np
import gmpy
import pandas as pd
import scipy

from typing import List

from sop_cost_model import SOPCostModel
from sop_utils import Codelet, Acc
from sbench.loaders.dlmc import DLMCLoader
from sbench.loaders.load import load_dense
from sop_utils import tile_matrix

from sop_hand_configs import sop4_strategies, sop6_strategies, sop8_strategies
import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


def pattern_code(vec):
    vec = vec.to(dtype=torch.int)
    pat = 0
    for idx, i in enumerate(vec):
        pat |= i.item() << idx
    return pat


def convert_panel_to_codes(A):
    pat_codes = torch.zeros(A.shape[0], dtype=torch.int)

    for i, row in enumerate(A):
        code = pattern_code(row)
        pat_codes[i] = code

    return pat_codes


RUNS_TO_MEDIAN = 15


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
        width = 64
    else:
        width = 512
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


def random_sp_matrix(shape, sparsity):
    return (torch.rand(shape) <= (1 - sparsity)).to(dtype=torch.float)


def rand_matrices():
    dir_name = "/home/lwilkinson/Downloads/rand_probs"
    list_of_paths = os.listdir(dir_name)
    # for path in list_of_paths:
    #     matrix, dic_mat, sop_height, wdt = read_unit_codelet_probability(os.path.join(dir_name, path))
    #     yield torch.Tensor(matrix), path, int(path.split("_")[3].strip("%"))/100

    # tile_shape = [48, 128]
    # tile_to_test = [
    #     ('random (70%)', random_sp_matrix(shape=tile_shape, sparsity=0.7), 0.7),
    #     ('random (80%)', random_sp_matrix(shape=tile_shape, sparsity=0.8), 0.8),
    #     ('random (90%)', random_sp_matrix(shape=tile_shape, sparsity=0.9), 0.9),
    #     ('random (95%)', random_sp_matrix(shape=tile_shape, sparsity=0.95), 0.95),
    # ]
    #
    # for name, tile, sparsity in tile_to_test:
    #     yield tile, name, sparsity

    tile_shape = [48, 64]
    tile_to_test = [
        ('random (70%)', random_sp_matrix(shape=tile_shape, sparsity=0.7), 0.7),
        ('random (80%)', random_sp_matrix(shape=tile_shape, sparsity=0.8), 0.8),
        ('random (90%)', random_sp_matrix(shape=tile_shape, sparsity=0.9), 0.9),
        ('random (95%)', random_sp_matrix(shape=tile_shape, sparsity=0.95), 0.95),
    ]

    for name, tile, sparsity in tile_to_test:
        yield tile, name, sparsity


def ilp_mappings(dir_name, filter):
    #dir_name = "/home/lwilkinson/Downloads/nano4_70_95"
    list_of_paths = os.listdir(dir_name)
    for path in list_of_paths:
        if filter in path and 'txt' in path:
            print("testing {}".format(path))
            patterns, mapping = file_dic_todic(os.path.join(dir_name, path))
            print(patterns, mapping)
            mapping.update({0: 0})
            yield patterns, mapping, path


if __name__ == "__main__":
    csv_rows = []
    bCols = 64

    for matrix, mtx_path, sparsity in rand_matrices():
        if matrix.shape[0] == 4:
            matrix = matrix.repeat(2, 1)
        print(matrix.shape)
        sp_str = "random_" + str(int(sparsity * 100))
        print(sp_str)

        csv_row_params = {
            "bCols": bCols,
            "name": mtx_path,
            "tileM": matrix.shape[0],
            "tileN": matrix.shape[1]
        }
        B = torch.ones((matrix.shape[1], bCols))
        sol = matrix @ B

        def run_patterns(_patterns, _mapping, name, path):
            acc_N = min(bCols // 16, 2)
            module = sop_driver.make_sop_module(Acc(4, acc_N), _patterns, _mapping)
            print(module.kernel_id)
            sop_tile = module.make_sop_tile(matrix)

            times = []
            for run in range(RUNS_TO_MEDIAN):
                time, result = module.execute_tile(sop_tile, B)
                times.append(time)
            time = np.median(np.array(times))

            print("SOP4", name, len(_patterns), mtx_path, time, sop_tile.padding(), torch.allclose(result, sol))

            csv_rows.append({
                "method": "SOP4",
                "submethod": "SOP4 " + name,
                "time": time,
                "correct": torch.allclose(result, sol),
                "padding": sop_tile.padding(),
                "num_patterns": len(_patterns),
                **csv_row_params
            })

        #print("SOP4_" + sp_str)
        for patterns, mapping, path in ilp_mappings("/home/lwilkinson/Downloads/nano4_70_95", sp_str):
            run_patterns(patterns, lambda x: mapping[x], "ILP")
        # for config in sop4_strategies:
        #     if len(config.all_patterns()) == 15: continue
        #     run_patterns(config.all_patterns(), config.map_pattern, "Hand")
        # run_patterns(list(range(1, 2**4)), lambda x: x, "All Patterns")

        def run_patterns(_patterns, _mapping, name):
            acc_N = min(bCols // 16, 2)
            module = sop_driver.make_sop_module(Acc(8, acc_N), _patterns, _mapping)
            sop_tile = module.make_sop_tile(matrix)

            times = []
            for run in range(RUNS_TO_MEDIAN):
                time, result = module.execute_tile(sop_tile, B)
                times.append(time)
            time = np.median(np.array(times))

            print("SOP8", name, mtx_path, time, len(_patterns), sop_tile.padding(), torch.allclose(result, sol))
            csv_rows.append({
                "method": "SOP8",
                "submethod": "SOP8 " + name,
                "time": time,
                "correct": torch.allclose(result, sol),
                "padding": sop_tile.padding(),
                "num_patterns": len(_patterns),
                **csv_row_params
            })

        for patterns, mapping, path in ilp_mappings("/home/lwilkinson/Downloads/nano8_70_95", sp_str):
            run_patterns(patterns, lambda x: mapping[x], "ILP")
        for config in sop8_strategies:
            if len(config.all_patterns()) == 255: continue
            run_patterns(config.all_patterns(), config.map_pattern, "Hand")
        #
        # run_patterns([i for i in range(1, 2**8)], lambda x: x, "All Patterns")


df = pd.DataFrame(csv_rows)
df.to_csv('sop_bench_ilp.csv')
print(df)


#
# def random_sp_matrix(shape, sparsity):
#     return (torch.rand(shape) <= (1 - sparsity)).to(dtype=torch.float)
#
#
# def run_all_sop4(matrix, bCols, csv_row_params):
#     acc_N = min(bCols // 16, 2)
#     csv_rows = []
#     sol = matrix @ B
#
#     for config in sop4_strategies:
#         module = sop_driver.make_sop_module(Acc(4, acc_N), config.all_patterns(), config.map_pattern)
#         sop_tile = module.make_sop_tile(matrix)
#         time, result = module.execute_tile(sop_tile, B)
#         print("SOP4", len(config.all_patterns()),name, time, sop_tile.padding(), torch.allclose(result, sol))
#
#         csv_rows.append({
#             "method": "SOP4",
#             "time": time,
#             "correct": torch.allclose(result, sol),
#             "padding": sop_tile.padding(),
#             "num_patterns": len(config.all_patterns()),
#             **csv_row_params
#         })
#
#     return csv_rows
#
#
# def run_all_sop6(matrix, bCols, csv_row_params):
#     acc_N = min(bCols // 16, 2)
#     csv_rows = []
#     sol = matrix @ B
#
#     for config in sop6_strategies:
#         module = sop_driver.make_sop_module(Acc(6, acc_N), config.all_patterns(), config.map_pattern)
#         sop_tile = module.make_sop_tile(matrix)
#         time, result = module.execute_tile(sop_tile, B)
#         print("SOP6", len(config.all_patterns()), name, time, sop_tile.padding(), torch.allclose(result, sol))
#
#         csv_rows.append({
#             "method": "SOP6",
#             "time": time,
#             "correct": torch.allclose(result, sol),
#             "padding": sop_tile.padding(),
#             "num_patterns": len(config.all_patterns()),
#             **csv_row_params
#         })
#
#     return csv_rows
#
#
# def run_all_sop8(matrix, bCols, csv_row_params):
#     acc_N = min(bCols // 16, 2)
#     csv_rows = []
#     sol = matrix @ B
#
#     for config in sop8_strategies:
#         module = sop_driver.make_sop_module(Acc(8, acc_N), config.all_patterns(), config.map_pattern)
#         sop_tile = module.make_sop_tile(matrix)
#         time, result = module.execute_tile(sop_tile, B, 1024)
#         print("SOP8", len(config.all_patterns()), name, time, sop_tile.padding(), torch.allclose(result, sol))
#
#         csv_rows.append({
#             "method": "SOP8",
#             "time": time,
#             "correct": torch.allclose(result, sol),
#             "padding": sop_tile.padding(),
#             "num_patterns": len(config.all_patterns()),
#             **csv_row_params
#         })
#
#     return csv_rows
#
#
# tile_shape = [48, 32]
# tile_to_test = [
#     ('random (70%)', random_sp_matrix(shape=tile_shape, sparsity=0.7)),
#     ('random (75%)', random_sp_matrix(shape=tile_shape, sparsity=0.75)),
#     ('random (80%)', random_sp_matrix(shape=tile_shape, sparsity=0.8)),
#     ('random (85%)', random_sp_matrix(shape=tile_shape, sparsity=0.85)),
#     ('random (90%)', random_sp_matrix(shape=tile_shape, sparsity=0.9)),
#     ('random (95%)', random_sp_matrix(shape=tile_shape, sparsity=0.95)),
# ]
#
# bcols_to_test = [16, 32, 64, 128]
# csv_rows = []
#
# for bcols in bcols_to_test:
#     B = torch.ones((tile_shape[1], bcols))
#     for name, tile in tile_to_test:
#         csv_row_params = {
#             "bCols": bcols,
#             "name": name,
#             "tileM": tile_shape[0],
#             "tileN": tile_shape[1]
#         }
#
#         print("=>", name, "csr", bcols)
#         time, result = sop_driver.csr.executor(tile.to_sparse_csr(), B, 1024, min(bcols, 32))
#         print(time)
#
#         csv_rows.append({
#             "method": "CSR",
#             "time": time,
#             "correct": torch.allclose(result, tile @ B),
#             "padding": 0,
#             "num_patterns": 0,
#             **csv_row_params
#         })
#
#         csv_rows += run_all_sop4(tile, bcols, csv_row_params)
#         csv_rows += run_all_sop6(tile, bcols, csv_row_params)
#         csv_rows += run_all_sop8(tile, bcols, csv_row_params)
#