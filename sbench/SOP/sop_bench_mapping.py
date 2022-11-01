import math
import torch
import sop_driver
import numpy as np
import gmpy
import pandas as pd
import hashlib

from typing import List

from sop_cost_model import SOPCostModel
from sop_utils import Codelet, Acc
from sop_hand_configs import sop4_strategies, sop6_strategies, sop8_strategies
from sbench.loaders.dlmc import DLMCLoader
from sbench.loaders.load import load_dense
from sop_utils import tile_matrix

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


RUNS_TO_AVERAGE = 15

dlmc_loader = DLMCLoader(file_list=SCRIPT_DIR + "/../../tools/filelists/transformer_magnitude_80.txt",
                         loader=load_dense)

torch.set_printoptions(precision=2, edgeitems=8, linewidth=220)


def random_sp_matrix(shape, sparsity):
    return (torch.rand(shape) <= (1 - sparsity)).to(dtype=torch.float)


def save_mapping(M_r, mapping: callable):
    mapping_dict = {}
    for i in range(1, 2**M_r):
        mapping_dict[i] = mapping(i)

    id = hashlib.md5(str(mapping_dict).encode('utf-8')).hexdigest()[-5:]

    print("Saving mapping", id)
    with open(SCRIPT_DIR + "/mappings/mapping_{}.txt".format(id), "w+") as f:
        f.write("{}\n".format(M_r))
        for k, v in mapping_dict.items():
            f.write("{}: {}\n".format(k, v))

    return id


TRIAL_ITERATIONS = 32


def run_all_sop4(matrix, bCols, csv_row_params):
    acc_N = min(bCols // 16, 2)
    csv_rows = []
    sol = matrix @ B

    for config in sop4_strategies:
        module = sop_driver.make_sop_module(Acc(4, acc_N), config.all_patterns(), config.map_pattern)
        sop_tile = module.make_sop_tile(matrix)
        time, result = module.execute_tile(sop_tile, B, TRIAL_ITERATIONS)
        print("SOP4", len(config.all_patterns()),name, time, sop_tile.padding(), torch.allclose(result, sol))

        csv_rows.append({
            "method": "SOP4",
            "mapping": save_mapping(4, config.map_pattern),
            "time": time,
            "correct": torch.allclose(result, sol),
            "padding": sop_tile.padding(),
            "num_patterns": len(config.all_patterns()),
            **csv_row_params
        })

    return csv_rows


def run_all_sop6(matrix, bCols, csv_row_params):
    acc_N = min(bCols // 16, 2)
    csv_rows = []
    sol = matrix @ B

    for config in sop6_strategies:
        module = sop_driver.make_sop_module(Acc(6, acc_N), config.all_patterns(), config.map_pattern)
        print("Running ", module.kernel_id)
        sop_tile = module.make_sop_tile(matrix)
        time, result = module.execute_tile(sop_tile, B, TRIAL_ITERATIONS)
        print("SOP6", len(config.all_patterns()), name, time, sop_tile.padding(), torch.allclose(result, sol))

        csv_rows.append({
            "method": "SOP6",
            "mapping": save_mapping(6, config.map_pattern),
            "time": time,
            "correct": torch.allclose(result, sol),
            "padding": sop_tile.padding(),
            "num_patterns": len(config.all_patterns()),
            **csv_row_params
        })

    return csv_rows


def run_all_sop8(matrix, bCols, csv_row_params):
    acc_N = min(bCols // 16, 2)
    csv_rows = []
    sol = matrix @ B

    for config in sop8_strategies:
        module = sop_driver.make_sop_module(Acc(8, acc_N), config.all_patterns(), config.map_pattern)
        print("Running ", module.kernel_id)
        sop_tile = module.make_sop_tile(matrix)
        time, result = module.execute_tile(sop_tile, B, TRIAL_ITERATIONS)
        print("SOP8", len(config.all_patterns()), name, time, sop_tile.padding(), torch.allclose(result, sol))

        csv_rows.append({
            "method": "SOP8",
            "mapping": save_mapping(8, config.map_pattern),
            "time": time,
            "correct": torch.allclose(result, sol),
            "padding": sop_tile.padding(),
            "num_patterns": len(config.all_patterns()),
            **csv_row_params
        })

    return csv_rows


tile_shape = [48, 256]
tile_to_test = [
    ('random (70%)', random_sp_matrix(shape=tile_shape, sparsity=0.7)),
    ('random (75%)', random_sp_matrix(shape=tile_shape, sparsity=0.75)),
    ('random (80%)', random_sp_matrix(shape=tile_shape, sparsity=0.8)),
    ('random (85%)', random_sp_matrix(shape=tile_shape, sparsity=0.85)),
    ('random (90%)', random_sp_matrix(shape=tile_shape, sparsity=0.9)),
    ('random (95%)', random_sp_matrix(shape=tile_shape, sparsity=0.95)),
]

bcols_to_test = [128]
csv_rows = []

for bcols in bcols_to_test:
    B = torch.ones((tile_shape[1], bcols))
    for name, tile in tile_to_test:
        csv_row_params = {
            "bCols": bcols,
            "name": name,
            "tileM": tile_shape[0],
            "tileN": tile_shape[1]
        }

        print("=>", name, "csr", bcols)
        time, result = sop_driver.csr.executor(tile.to_sparse_csr(), B, 1024, min(bcols, 64))
        print(time)

        csv_rows.append({
            "method": "CSR",
            "time": time,
            "correct": torch.allclose(result, tile @ B),
            "padding": 0,
            "num_patterns": 0,
            **csv_row_params
        })

        csv_rows += run_all_sop4(tile, bcols, csv_row_params)
        csv_rows += run_all_sop6(tile, bcols, csv_row_params)
        csv_rows += run_all_sop8(tile, bcols, csv_row_params)


df = pd.DataFrame(csv_rows)
df.to_csv('sop_bench.csv')
print(df)
