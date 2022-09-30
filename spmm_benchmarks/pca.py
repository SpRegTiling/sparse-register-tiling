import numpy as np
import torch
import inspect
import sys
import altair as alt
import pandas as pd
import json
import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
import itertools

from torch.profiler import profile, ProfilerActivity
from altair_saver import save
from collections import defaultdict

from xformers.sparse.csr_tensor import SparseCSRTensor
from xformers.sparse.utils import _csr_to_coo

from spmm_benchmarks.utils.test import construct_test_tensors
from spmm_benchmarks.utils.plot import sanitize_path_string
from spmm_benchmarks.loaders.dlmc import DLMCLoader
from spmm_benchmarks.pytorch_profiler.wrapper import profile_func

from collections import namedtuple

OpInfo = namedtuple("OpInfo", ["storage_type", "op", "profiled_name", "plot_name", "tune"])


##
#   Profiling Script
##

if __name__ == '__main__':
    loader = DLMCLoader(models=['rn50'])

#    for shape, row_ptrs, col_indices, name in DLMCLoader():
    for shape, row_ptrs, col_indices, name in loader:

        print(name)
        weight = {}
        golden = {}
        nnz = len(col_indices)

        values = torch.ones(nnz)

        # Unsqueezing since the spmm is batched
        weight["csr"] = SparseCSRTensor(
            row_offsets=row_ptrs,
            column_indices=col_indices,
            values=values.unsqueeze(0),
            shape=(1, *shape)
        )

        weight["dense"] = SparseCSRTensor._to_dense(weight["csr"])
        print(weight["dense"].shape)

        pca = torch.pca_lowrank(weight["dense"], niter=100)
        print(len(pca))
        directions = pca[-1].squeeze(0).transpose(-1, -2)
        print(directions.shape)
        print(directions[0])
        topk = torch.topk(abs(directions[0]), 8)
        print(topk.indices)

        pat = torch.zeros(len(directions[0]))
        pat = pat.scatter(0, topk.indices, 1)
        torch.set_printoptions(profile="full")
        print(pat)
        for row in weight["dense"][0]:
            #print(row * pat)
            matches = (row * pat).sum()
            if matches > 6:
                print(matches)
        import sys; sys.exit()

