import math
import torch
import sop_driver
import numpy as np
import gmpy
import pandas as pd
import altair as alt

from typing import List

from sop_cost_model import SOPCostModel
from sop_utils import Codelet, Acc
from spmm_benchmarks.loaders.dlmc import DLMCLoader
from spmm_benchmarks.loaders.load import load_dense
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


class SOPOrig:
    def __init__(self, merge_patterns, split_patterns, split_cost=1.1, merge_cost=1, min_merge=2, recursive=False):
        self.merge_patterns = merge_patterns
        self.split_patterns = split_patterns
        self.split_cost = split_cost
        self.merge_cost = merge_cost
        self.min_merge = min_merge
        self.recursive = recursive

    def all_patterns(self):
        return self.merge_patterns + self.split_patterns

    def map_pattern(self, pat_code, cant_overlap=0):
        if pat_code == 0: return [0]

        pat_code = int(pat_code)
        if gmpy.popcount(pat_code) < self.min_merge:
            return [pat_code]

        best_pattern = None
        min_cost = torch.inf

        patterns = []

        for pattern in self.merge_patterns:
            if pattern & cant_overlap > 0: continue

            padding = gmpy.popcount(pattern) - gmpy.popcount(pattern & pat_code)
            split = gmpy.popcount(~pattern & pat_code)
            cost = self.split_cost * split + self.merge_cost * padding
            if cost < min_cost:
                best_pattern = pattern
                min_cost = cost

        assert best_pattern is not None

        patterns.append(best_pattern)

        split_pattern = ~best_pattern & pat_code

        if split_pattern:
            if self.recursive:
                #print(split_pattern, self.map_pattern(split_pattern))
                patterns += self.map_pattern(split_pattern, cant_overlap=best_pattern | cant_overlap)
            else:
                idx = 0
                while split_pattern:
                    if split_pattern & 1:
                        patterns.append(1 << idx)

                    idx += 1
                    split_pattern >>= 1

        #if split_pattern

        # if len(patterns) > 2:
        #     print("multisplit", self, self.recursive, patterns)

        return patterns


def gen_all_combos(vec_height, nnz):
    if nnz == 1:
        return [1 << i for i in range(vec_height)]

    combos = set()
    for pat in gen_all_combos(vec_height, nnz-1):
        for pat1 in gen_all_combos(vec_height, 1):
            if gmpy.popcount(pat | pat1) == nnz:
                combos.add(pat | pat1)

    return list(combos)


sop4_strategies = [
    SOPOrig(
        merge_patterns=[],
        split_patterns=list(range(1, 2**4)),
        split_cost=10000,
        min_merge=5
    ),
#     SOPOrig(
#         merge_patterns=gen_all_combos(4, 2) + [0b00001111],
#         split_patterns=[],
#         split_cost=10000,
#         min_merge=1
#     ),
#     SOPOrig(
#         merge_patterns=[0b00001111],
#         split_patterns=[1 << i for i in range(4)],
#         split_cost=10000
#     ),
#     SOPOrig(
#         merge_patterns=gen_all_combos(4, 3) + [0b00001111],
#         split_patterns=[1 << i for i in range(4)],
#         split_cost=10000
#     ),
#     SOPOrig(
#         merge_patterns=gen_all_combos(4, 2) + [0b00001111],
#         split_patterns=[1 << i for i in range(4)],
#         split_cost=10000
#     )
]

sop6_strategies = [
    SOPOrig(
        merge_patterns=[],
        split_patterns=list(range(1, 2**6)),
        split_cost=10000,
        min_merge=7
    ),
    SOPOrig(
        merge_patterns=[
            0b010101, 0b101010,
            0b111000, 0b000111,
            0b111111
        ],
        split_patterns=[1 << i for i in range(6)],
        split_cost=10000
    ),
    SOPOrig(
        merge_patterns=[
            0b000011, 0b001100, 0b110000, 0b101010, 0b010101,
            0b111111
        ],
        split_patterns=[1 << i for i in range(6)],
        split_cost=2.1,
        recursive=True
    )
]


sop8_strategies = [
    SOPOrig(
        merge_patterns=[],
        split_patterns=list(range(1, 256)),
        split_cost=10000,
        min_merge=9
    ),
    SOPOrig(
        merge_patterns=[0b11111111],
        split_patterns=[1 << i for i in range(8)],
        split_cost=10000
    ),
    SOPOrig(
        merge_patterns=[
            0b01010101, 0b10101010, 0b11000011,
            0b00111100,
            0b11010111, 0b10111101,
            0b11111111
        ],
        split_patterns=[1 << i for i in range(8)]
    ),
    SOPOrig(
        merge_patterns=[
            0b01010101, 0b10101010, 0b11000011,
            0b00111100, 0b00001111, 0b11110000,
            0b11111100, 0b11110011, 0b11001111, 0b00111111,
            0b11111111
        ],
        split_patterns=[1 << i for i in range(8)],
        split_cost=10000
    ),
    SOPOrig(
        merge_patterns=[
            0b01010101, 0b10101010, 0b11000011, 0b00111100, 0b00001111, 0b11110000,
            0b11111111
        ],
        split_patterns=[1 << i for i in range(8)],
        split_cost=10000
    ),
    SOPOrig(
        merge_patterns=gen_all_combos(8, 4) + gen_all_combos(8, 6) + [0b11111111],
        split_patterns=[1 << i for i in range(8)],
        split_cost=10000
    ),
    SOPOrig(
        merge_patterns=gen_all_combos(8, 3) + gen_all_combos(8, 5) + [0b11111111],
        split_patterns=[1 << i for i in range(8)],
        split_cost=10000
    ),
    SOPOrig(
        merge_patterns=gen_all_combos(8, 2) + gen_all_combos(8, 5) + [0b11111111],
        split_patterns=[1 << i for i in range(8)],
        split_cost=2.1,
        merge_cost=1,
        recursive=True
    ),
    SOPOrig(
        merge_patterns=[0b11111111],
        split_patterns=[1 << i for i in range(8)]
    ),
    SOPOrig(
        merge_patterns=[
            0b01010101, 0b10101010, 0b11000011, 0b00111100, 0b00001111, 0b11110000,
            0b11111111
        ],
        split_patterns=[1 << i for i in range(8)],
        split_cost=1.1,
        merge_cost=1,
        recursive=True
    ),
    SOPOrig(
        merge_patterns=[
            0b01010101, 0b10101010, 0b11000011, 0b00111100, 0b00001111, 0b11110000,
            0b11111100, 0b11110011, 0b11001111, 0b00111111,
            0b11111111
        ],
        split_patterns=[1 << i for i in range(8)],
        split_cost=1.1,
        merge_cost=1,
        recursive=True
    ),
    SOPOrig(
        merge_patterns=[
            0b00000011, 0b00001100, 0b00110000, 0b11000000,
            0b00111100, 0b00001111, 0b11110000,
            0b01010101, 0b10101010, 0b11111111],
        split_patterns=[1 << i for i in range(8)],
        split_cost=2.1,
        recursive=True
    ),
    SOPOrig(
        merge_patterns=[
            0b00000011, 0b00001100, 0b00110000, 0b11000000,
            0b00011111, 0b11111000,
            0b01010101, 0b10101010, 0b11111111],
        split_patterns=[1 << i for i in range(8)],
        split_cost=1.1,
        recursive=True
    ),
    SOPOrig(
        merge_patterns=gen_all_combos(8, 2) + [
            0b01010101, 0b10101010,
            #0b00001111, 0b00111100, 0b11110000,
            0b11111111
        ],
        split_patterns=[1 << i for i in range(8)],
        split_cost=3.1,
        merge_cost=1,
        recursive=True
    ),
    SOPOrig(
        merge_patterns=[
            0b00000011, 0b00001100, 0b00110000, 0b11000000,
            0b11111100, 0b11110011, 0b11001111, 0b00111111,
            0b01010101, 0b10101010, 0b11111111],
        split_patterns=[1 << i for i in range(8)],
        split_cost=3.1,
        recursive=True
    )
]

dlmc_loader = DLMCLoader(file_list=SCRIPT_DIR + "/../../tools/filelists/transformer_magnitude_80.txt",
                         loader=load_dense)

torch.set_printoptions(precision=2, edgeitems=8, linewidth=220)


def random_sp_matrix(shape, sparsity):
    return (torch.rand(shape) <= (1 - sparsity)).to(dtype=torch.float)


def run_all_sop4(matrix, bCols, bStride, csv_row_params):
    acc_N = max(min(bStride // 16, 2), 1)
    print(acc_N)
    csv_rows = []
    sol = matrix @ B

    for config in sop4_strategies:
        module = sop_driver.make_sop_module(Acc(4, acc_N), config.all_patterns(), config.map_pattern)
        sop_tile = module.make_sop_tile(matrix)
        time, result = module.execute_tile(sop_tile, B, N_c=bCols)
        if not torch.allclose(result, sol):
            print(result)
            print(sol)
        print("SOP4", len(config.all_patterns()),name, time, sop_tile.padding(), torch.allclose(result, sol))

        csv_rows.append({
            "method": "SOP4",
            "time": time,
            "correct": torch.allclose(result, sol),
            "padding": sop_tile.padding(),
            "num_patterns": len(config.all_patterns()),
            **csv_row_params
        })

    return csv_rows


def run_all_sop6(matrix, bCols, csv_row_params):
    acc_N = max(min(bCols // 16, 2), 1)
    csv_rows = []
    sol = matrix @ B

    for config in sop6_strategies:
        module = sop_driver.make_sop_module(Acc(6, acc_N), config.all_patterns(), config.map_pattern)
        sop_tile = module.make_sop_tile(matrix)
        time, result = module.execute_tile(sop_tile, B)
        print("SOP6", len(config.all_patterns()), name, time, sop_tile.padding(), torch.allclose(result, sol))

        csv_rows.append({
            "method": "SOP6",
            "time": time,
            "correct": torch.allclose(result, sol),
            "padding": sop_tile.padding(),
            "num_patterns": len(config.all_patterns()),
            **csv_row_params
        })

    return csv_rows


def run_all_sop8(matrix, bCols, bStride, csv_row_params):
    acc_N = max(min(bStride // 16, 2), 1)
    csv_rows = []
    sol = matrix @ B

    for config in sop8_strategies:
        module = sop_driver.make_sop_module(Acc(8, acc_N), config.all_patterns(), config.map_pattern)
        sop_tile = module.make_sop_tile(matrix)
        time, result = module.execute_tile(sop_tile, B, 1024, N_c=bCols)
        print("SOP8", len(config.all_patterns()), name, time, sop_tile.padding(), torch.allclose(result, sol))

        csv_rows.append({
            "method": "SOP8",
            "time": time,
            "correct": torch.allclose(result, sol),
            "padding": sop_tile.padding(),
            "num_patterns": len(config.all_patterns()),
            **csv_row_params
        })

    return csv_rows


tile_shape = [48, 64]
tile_to_test = [
    ('random (70%)', random_sp_matrix(shape=tile_shape, sparsity=0.7)),
    # ('random (75%)', random_sp_matrix(shape=tile_shape, sparsity=0.75)),
    # ('random (80%)', random_sp_matrix(shape=tile_shape, sparsity=0.8)),
    # ('random (85%)', random_sp_matrix(shape=tile_shape, sparsity=0.85)),
    # ('random (90%)', random_sp_matrix(shape=tile_shape, sparsity=0.9)),
    # ('random (95%)', random_sp_matrix(shape=tile_shape, sparsity=0.95)),
]

bcols_to_test = [
    (64, 64),
    # (1, 1),
    # (16, 16),
    # (18, 18),
    # (32, 32),
    # (34, 34),
    # (34, 64),
    # (64, 64),
    # (192, 192),
    # (204, 204),
    # (204, 208),
    # (208, 208),
    # (784, 784),
    # (800, 800)
] # 16, 20, 64, 80,
csv_rows = []

for bcols, bstride in bcols_to_test:
    print("BCols", bcols)
    B = torch.ones((tile_shape[1], bstride))
    B[:,bcols:] = 0

    for name, tile in tile_to_test:
        csv_row_params = {
            "bCols": bcols,
            "bStride": bstride,
            "name": name,
            "tileM": tile_shape[0],
            "tileN": tile_shape[1]
        }
        csv_rows += run_all_sop4(tile, bcols, bstride, csv_row_params)
        # csv_rows += run_all_sop6(tile, bcols, csv_row_params)
        # csv_rows += run_all_sop8(tile, bcols, csv_row_params)


df = pd.DataFrame(csv_rows)
df.to_csv('sop_masked_bench.csv')
print(df)
