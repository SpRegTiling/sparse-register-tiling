import itertools
import torch

from functools import partial
from torch.autograd.profiler import record_function
from torch.utils import benchmark

from xformers import _is_sparse_available
from xformers.components.attention.core import SparseCS, _sparse_bmm

from sbench.pytorch_profiler.run import torch_profiler
from sbench.utils.test import construct_test_tensors
from sbench.utils.plot import stack_plot, speedup_plot_torch_benchmark, stack_area_plot_torch_profiler
from sbench.full_bw_fw_pass.linear import *


MIN_RUN_TIME = 0.5
SHAPES = [[8, 8], [256, 1024], [128, 256]]
SPARSITIES = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

PROFILER = "torch_profiler"
PLOT_TYPE = "area-stack"


##
#   Profiler Definitions
##


def torch_benchmark(task_spec):
    return benchmark.Timer(**task_spec).blocked_autorange(min_run_time=MIN_RUN_TIME)


dense = {
    "forward": {
        "children": {
            "aten::matmul": "fw bmm"
        },
        "remaining": "fw remaining"
    },
    "*": [{
        "prefix": "autograd::engine::evaluate_function:",
        "remaining": "bw remaining",
        "children": {
            "BmmBackward0": {
                "children": {
                    "aten::bmm": "bw bmm",
                    "aten::transpose": "bw transpose",
                },
                "remaining": "bw remaining"
            }
        }
    }]
}


spmm = {
    "forward": {
        "children": {
            "_spmm": "fw spmm"
        },
        "remaining": "fw remaining"
    },
    "*": [{
        "prefix": "autograd::engine::evaluate_function:",
        "remaining": "bw remaining",
        "children": {
            "_spmmBackward": {
                "children": {
                    "xformers::sddmm_sputnik": "bw sddmm",
                    "xformers::spmm_sputnik": "bw spmm"
                },
                "remaining": "bw transpose"
            }
        }
    }]
}

profiler_runners = {
    "torch_profiler": {
        "sputnik": partial(torch_profiler, profile_mapping=spmm),
        "dense": partial(torch_profiler, profile_mapping=dense),
    },
    "torch_benchmark": {
        "sputnik": torch_benchmark,
        "dense": torch_benchmark,
    }
}

result_mapper = {
    "stack": {
        "torch_profiler": lambda x: x,
    },
    "speed-up": {
        "torch_benchmark": lambda x: benchmark.Compare(results),
    },
    "area-stack": {
        "torch_profiler": lambda x: x,
    }
}

plotter = {
    "stack": {
        "torch_profiler": stack_plot,
    },
    "speed-up": {
        "torch_benchmark": speedup_plot_torch_benchmark,
    },
    "area-stack": {
        "torch_profiler": stack_area_plot_torch_profiler,
    }
}


##
#   Test cases
##

def profile_bmm_fw_bw_recompute_transpose(args: dict):
    from xformers.sparse import _csr_ops
    args["label"] = "full_pass"

    _csr_ops.FORCE_TRANSPOSE_RECOMPUTE = True
    results = run_sparse_benchmarks(**args)
    _csr_ops.FORCE_TRANSPOSE_RECOMPUTE = False

    return results


def profile_bmm_fw_bw_precompute_idx(args: dict):
    args["label"] = "full_pass_t_idx_precomp"
    return run_sparse_benchmarks(**args)


def profile_bmm_fw_bw_precompute_transpose(args: dict):
    args["label"] = "full_pass_t_precomp"
    args["precompute_transpose"] = True
    return run_sparse_benchmarks(**args)


if __name__ == '__main__':
    import copy

    print(f'Running {PROFILER} and creating a {PLOT_TYPE} plot')

    results = []

    default_args = dict(
        runner=profiler_runners[PROFILER],
        f="{}_linear_pre", setup=zero_grad,
        dense_types=["dense"],
        sparse_types=["sputnik"], sparsities=SPARSITIES,
        device="cuda"
    )

    for B, M, K in zip(*SHAPES):
        input_shape = (B, M, M)
        weight_shape = (B, K, M)

        default_args["input_shape"] = input_shape
        default_args["weight_shape"] = weight_shape
        default_args["description"] = f"B={B}, M={M}, K={K}"

        if PLOT_TYPE == "stacked":
            results += profile_bmm_fw_bw_recompute_transpose(copy.deepcopy(default_args))

        results += profile_bmm_fw_bw_precompute_idx(copy.deepcopy(default_args))

        if PLOT_TYPE == "stacked":
            results += profile_bmm_fw_bw_precompute_transpose(copy.deepcopy(default_args))

    results = result_mapper[PLOT_TYPE][PROFILER](results)
    plotter[PLOT_TYPE][PROFILER](results, title="Full Forward and Backwards Post Multiply")

    print("Done!")