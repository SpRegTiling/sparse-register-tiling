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
#   Utils
##


def _total_cpu_time(results, name):
    for event in results.key_averages():
        if event.key == name:
            return event.cpu_time_total
    return None


##
#   CSR Support
##

def csr_spmm_call(spmm, csr: SparseCSRTensor, input: torch.Tensor):
    assert isinstance(input, torch.Tensor)
    assert isinstance(csr, SparseCSRTensor)

    assert csr.ndim == 3
    assert input.ndim == 3

    b = input

    _, m, n = csr.shape
    row_indices = csr._csr_row_indices
    values = csr.values()
    row_offsets = csr._csr_row_offsets
    column_indices = csr._csr_column_indices

    return spmm(b, row_indices, values, row_offsets, column_indices, m)


_csr_spmm_profile = profile_func(csr_spmm_call)


def csr_spmm_profile(spmm_op_info: OpInfo, *args, **kwargs):
    out, profile_results = _csr_spmm_profile(spmm_op_info.op, *args, **kwargs)
    exec_time = _total_cpu_time(profile_results, spmm_op_info.profiled_name)

    if exec_time is None:
        raise Exception(f'Failed to find {spmm_op_info.profiled_name } in the profiler results')

    return out, exec_time


##
#   Generic Op Support
##

def generic_op_profile(spmm_op_info: OpInfo, *args, **kwargs):
    def call_op(op, a: torch.Tensor, b: torch.Tensor):
        return op(a, b)

    profile_op = profile_func(call_op)
    out, profile_results = profile_op(spmm_op_info.op, *args, **kwargs)
    exec_time = _total_cpu_time(profile_results, spmm_op_info.profiled_name)

    if exec_time is None:
        print(profile_results.key_averages())
        raise Exception(f'Failed to find {spmm_op_info.profiled_name } in the profiler results')

    return out, exec_time


##
#   torch CSR Op Support
##

def torch_csr_spmm(csr: torch.Tensor, input: torch.Tensor):
    assert input.shape[0] == 1
    return csr.matmul(input.squeeze(0))


##
#   torch CSR Op Support
##

def torch_coo_spmm(coo: torch.Tensor, input: torch.Tensor):
    assert input.shape[0] == 1
    return torch.smm(coo, input.squeeze(0))


##
#   Plot
##

def save_plot(path, name, suffix, chart):
    path = path + "/" + f'{name}_{suffix}'
    if not os.path.exists(path): os.makedirs(path)

    # Sanitize the filename
    filename = f'{name}_{suffix}' + "_speed-up"
    filename = sanitize_path_string(filename)

    filepath = path + "/" + filename + '.svg'
    print('Saving:', filepath)
    save(chart, filepath)

    from wand.image import Image
    png_filepath = filepath.replace('.svg', '.png')
    with open(filepath) as svg_file, Image(blob=svg_file.read().encode('utf-8'), format="svg") as image:
        print('Saving:', png_filepath)
        image.save(filename=png_filepath)


def plot_dense_vs(name, results, path: str = 'plots'):
    def find(element, json):
        keys = element.split('.')
        rv = json
        for key in keys:
            rv = rv[key]
        return rv

    for key, value in list(results.values())[0].items():
        if type(value) is not dict: continue
        if name in value.keys(): key_path = f'{key}.{name}'

    sparsities = [(1 - round(v["density"], 2)) for v in results.values()]
    nnzs = [v["nnz"] for v in results.values()]
    dense = [find("baseline.torch_dense", v) for v in results.values()]
    profiled = [find(key_path, v) for v in results.values()]

    d = {
        "sparsities": sparsities,
        "nnzs": nnzs,
        "dense": dense,
        "profiled": profiled,
        "speed-up": np.array(dense) / np.array(profiled),
    }

    df = pd.DataFrame(data=d)

    print(df)
    chart = alt.Chart(df).mark_circle(size=60).encode(
        #x=alt.X('sparsities', scale=alt.Scale(type="log")),
        x='sparsities',
        y='speed-up',
        #color='nnzs',
        color=alt.Color(
            "nnzs",
            scale=alt.Scale(
                scheme="redblue",
                reverse=True
            )
        )
        #tooltip=['Name', 'Origin', 'Horsepower', 'Miles_per_Gallon']
    )

    line = alt.Chart(pd.DataFrame({'y': [1]})).mark_rule().encode(y='y')
    chart = chart + line

    save_plot(path, name, "vs_dense", chart)


def plot_vs_fastest_baseline(key_path, name, results, path: str = 'plots'):
    def find(element, json):
        keys = element.split('.')
        rv = json
        for key in keys:
            rv = rv[key]
        return rv

    def fastest_baseline(result):
        best_time = torch.inf
        best_baseline = None
        for baseline, time in result["baseline"].items():
            if time < best_time:
                best_time = time
                best_baseline = baseline
        return best_time, best_baseline

    sparsities = [(1 - round(v["density"], 2)) for v in results.values()]
    nnzs = [v["nnz"] for v in results.values()]
    baseline_zipped = [fastest_baseline(v) for v in results.values()]
    baseline_time, baseline_name = zip(*baseline_zipped)
    profiled = [find(key_path, v) for v in results.values()]

    print(baseline_time)
    print(baseline_name)

    d = {
        "sparsities": sparsities,
        "nnzs": nnzs,
        "baseline_name": baseline_name,
        "profiled": profiled,
        "speed-up": np.array(baseline_time) / np.array(profiled),
    }

    df = pd.DataFrame(data=d)
    chart = alt.Chart(df, title=f'{name} speed-up over fastest baseline').mark_circle(size=60).encode(
        x='sparsities',
        y='speed-up',
        color=alt.Color("baseline_name")
    )

    line = alt.Chart(pd.DataFrame({'y': [1]})).mark_rule().encode(y='y')
    chart = chart + line

    save_plot(path, name, "vs_fastest_baseline", chart)


##
#   Plot
##

def tune_tile_size(csr: SparseCSRTensor, input: torch.Tensor):
    tile_sizes = [8, 16, 32, 64]
    profiled_spmm = profile_func(csr_spmm_call, wait=0, warmup=1, active=2)

    min_time = torch.inf
    best_tile_shape = []

    for tile_shape in itertools.product(tile_sizes, tile_sizes):
        torch.ops.sparse.spmm_tiling_tuned_config(tile_shape[0], tile_shape[1], 32)
        out, profile_results = profiled_spmm(torch.ops.sparse.spmm_tiling_tuned, csr, input)

        import time

        # Warm-Up
        csr_spmm_call(torch.ops.sparse.spmm_tiling_tuned, csr, input)

        start = time.time()
        for i in range(3):
            csr_spmm_call(torch.ops.sparse.spmm_tiling_tuned, csr, input)
        time = time.time() - start

        if time < min_time:
            min_time = time
            best_tile_shape = tile_shape

    torch.ops.sparse.spmm_tiling_tuned_config(best_tile_shape[0], best_tile_shape[1], 32)
    print("best tileshape", best_tile_shape)


##
#   Profiling Script
##

if __name__ == '__main__':
    override_cache = True
    models = ['rn50']
    loader = DLMCLoader(models=models, pruning_methods=['magnitude_pruning'], sparsities=['0.8'])
    loader_hash = loader.deterministic_hash()


    def epilog(results):
        plot_dense_vs("torch_csr", results)
        plot_vs_fastest_baseline("profiled.spmm_tiling_tuned", f'tuned_tile_{"_".join(models)}', results)

    cache_file = f'{SCRIPT_DIR}/.cache/dlmc_bench_{loader_hash}.json'
    cached = os.path.exists(cache_file) if not override_cache else False

    if cached:
        with open(cache_file) as fp:
            results = json.load(fp)

        epilog(results)
        sys.exit()
    else:
        results = defaultdict(lambda: defaultdict(lambda: {}))


    # Used for validity check
    spmm_reference_op = OpInfo("csr", torch.ops.xformers.spmm_sputnik, "xformers::spmm_sputnik", "sputnik", None)

    spmm_ops_to_profile = [
        OpInfo("csr", torch.ops.sparse.spmm_tiling_tuned, "sparse::spmm_tiling_tuned", "spmm_tiling_tuned", tune_tile_size),
    ]

    spmm_ops_for_baseline = [
        OpInfo("dense", lambda a, b: a @ b, "aten::bmm", "torch_dense", None),
        OpInfo("torch_csr", torch_csr_spmm, "aten::matmul", "torch_csr", None),
        #OpInfo("torch_coo", torch_coo_spmm, "aten::sspaddmm", "torch_coo"),  # Always the slowest
    ]

    profile_op = {
        "csr": csr_spmm_profile,
        "dense": generic_op_profile,
        "torch_coo": generic_op_profile,
        "torch_csr": generic_op_profile,
    }

    BATCH_DIM = 256

#    for shape, row_ptrs, col_indices, name in DLMCLoader():
    for mat, name in loader:
        print(name)
        weight = {}
        golden = {}

        # Torch CSR does not support 3D tensors
        weight["torch_csr"] = mat
        # Unsqueezing since the spmm is batched
        weight["csr"] = SparseCSRTensor(
            mat.crow_indices(), mat.col_indices(), mat.values().unsqueeze(0), (1, *mat.shape))
        weight["dense"] = SparseCSRTensor._to_dense(weight["csr"])

        #
        # coo_row_indices, coo_col_indices = _csr_to_coo(*shape, row_ptrs, col_indices)
        # indices = torch.cat((coo_row_indices, coo_col_indices)).view(-1, nnz)
        # weight["torch_coo"] = torch.sparse_coo_tensor(indices, values, shape)

        input = construct_test_tensors("dense", (1, mat.shape[1], BATCH_DIM), device="cpu")

        assert spmm_reference_op.storage_type == "csr"
        golden = csr_spmm_call(spmm_reference_op.op, weight["csr"], input)

        results[name]["nnz"] = mat._nnz()
        results[name]["density"] = mat._nnz() / np.prod(mat.shape)

        for spmm_op in spmm_ops_to_profile:
            storage_type = spmm_op.storage_type

            if spmm_op.tune is not None:
                spmm_op.tune(weight[storage_type], input)

            out, exec_time = profile_op[storage_type](spmm_op, weight[storage_type], input)
            print("profiled", spmm_op.plot_name, exec_time)

            results[name]["profiled"][spmm_op.plot_name] = exec_time

            if not torch.allclose(out, golden):
                print("")

            assert torch.allclose(out, golden)

        for spmm_op in spmm_ops_for_baseline:
            storage_type = spmm_op.storage_type
            out, exec_time = profile_op[storage_type](spmm_op, weight[storage_type], input)
            if out.layout == torch.sparse_coo:
                out = out.to_dense()
            results[name]["baseline"][spmm_op.plot_name] = exec_time

            print("baseline", spmm_op.plot_name, exec_time)

            if not torch.allclose(out, golden):
                print("")

            assert torch.allclose(out, golden)

        del input, weight, out, golden

    print("/".join(cache_file.split("/")[:-1]) + "/")
    os.makedirs("/".join(cache_file.split("/")[:-1]) + "/", exist_ok=True)
    with open(cache_file, 'w+') as fp:
        fp.write(json.dumps(results))

    epilog(results)
