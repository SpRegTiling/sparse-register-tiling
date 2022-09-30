import math

import torch

import sop_driver
import numpy as np
import gmpy
import pandas as pd

from typing import List

from sop_cost_model import SOPCostModel, Codelet, Acc
from spmm_benchmarks.loaders.dlmc import DLMCLoader
from spmm_benchmarks.loaders.load import load_dense

import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

# SUPPORTED_PATTERNS_8 = [1 << i for i in range(8)] + [
#     0b01010101, 0b10101010,
#     0b11000011, 0b00111100, 0b00001111, 0b11110000,
#     0b11111100, 0b11110011, 0b11001111, 0b00111111,
#     0b11111111
# ]


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


RUNS_TO_AVERAGE = 50

cols = 64
B = torch.randn((cols, 32))
torch.set_printoptions(precision=2, edgeitems=8, linewidth=220)


class CodeletDict(dict):
    def __missing__(self, key):
        ret = self[key] = Codelet(nnz=key, cols=0)
        return ret


def get_codelets_from_panel(panel: torch.Tensor) -> List[Codelet]:
    codelets = CodeletDict()
    for row in panel.t():
        nnz = gmpy.popcount(pattern_code(row))
        if nnz == 0: continue
        codelets[nnz].cols += 1

    return list(codelets.values())


def build_synthetic_matrix_with_codelets(rows, codelets: Codelet):
    A = torch.zeros((rows, cols))
    curr_col_offset = 0
    for codelet in codelets:
        for j in range(codelet.cols):
            for i in range(codelet.nnz):
                A[i][curr_col_offset] = 1
            curr_col_offset += 1

    return A


def bench_cost_model(cost_model: SOPCostModel):
    configs_to_test = specific_cost_model.gen_sample_configs()

    samples = []
    for acc, codelets in configs_to_test:
        synthetic_panel = build_synthetic_matrix_with_codelets(acc.M, codelets)
        A = synthetic_panel.to_sparse_csr()

        patterns = [(1 << codelet.nnz) - 1 for codelet in codelets]
        module = sop_driver.make_sop_module(acc, patterns, lambda x: x)
        sop_tile = module.make_sop_tile(synthetic_panel)

        times = []
        for run in range(RUNS_TO_AVERAGE):
            time, result = module.execute_tile(sop_tile, B, 1024)
            times.append(time)

        samples.append((np.median(time), acc, codelets))

    specific_cost_model.solve_least_squares(samples)


def random_sp_matrix(shape, sparsity):
    return (torch.rand(shape) <= (1 - sparsity)).to(dtype=torch.float)


if __name__ == "__main__":

    specific_cost_model = SOPCostModel("ACC_SPECIFIC", Acc(4, 2))

    for test in range(3):
        bench_cost_model(specific_cost_model)
        print(specific_cost_model)

    configs_to_test = specific_cost_model.gen_sample_configs()

    samples = []
    predicted_cost = []
    for acc, codelets in configs_to_test:
        synthetic_panel = build_synthetic_matrix_with_codelets(acc.M, codelets)
        A = synthetic_panel.to_sparse_csr()

        patterns = [(1 << codelet.nnz) - 1 for codelet in codelets]
        module = sop_driver.make_sop_module(acc, patterns, lambda x: x)
        sop_tile = module.make_sop_tile(synthetic_panel)

        times = []
        for run in range(RUNS_TO_AVERAGE):
            time, result = module.execute_tile(sop_tile, B, 1024)
            times.append(time)

        samples.append(np.median(time))
        predicted_cost.append(specific_cost_model.cost_panel(codelets, acc))

    print(samples, predicted_cost)

    df = pd.DataFrame({"actual": samples, "predicted": predicted_cost})
    df.to_csv(SCRIPT_DIR + "/results/sop_mircobench.csv")

    chart = alt.Chart(
        df,
        title=[f'Cost Model Correlation']
    ).mark_circle().encode(
        x=alt.X('actual:Q'),
        y=alt.Y('predicted:Q')
    )

    chart.show()
    saver.save(chart, "cost_model_correlation.png", fmt="png", scale_fator=4)
