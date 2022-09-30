import numpy as np
import torch
import pandas as pd
import xformers
import matplotlib.pyplot as plt
import seaborn as sns
import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

from spmm_benchmarks.loaders.suitesparse import SuiteSparseLoader
from spmm_benchmarks.loaders.dlmc import DLMCLoader
from spmm_benchmarks.loaders.load import load_dense, load_csr, load_coo
from spmm_benchmarks.DDT import spmx_trace_gen, compute_consecutive_fopds, hist_mine_consecutive_fopds
from spmm_benchmarks.utils.cache import cache_dataframe, cached_return


PLOT_DIR = SCRIPT_DIR + "/../plots/"


ss_loader = SuiteSparseLoader()
dlmc_loader = DLMCLoader(models=['rn50'], pruning_methods=['magnitude_pruning'])


@cache_dataframe(refresh_cache=False)
def compute_runlengths():
    def _compute(loader, collection):
        avg = []
        max = []
        sparsities = []

        print(f'Computing runlengths for collection {collection}')
        for matrix, path in loader:
            print(path)

            avg_rl, min_rl, max_rl = torch.ops.spmm_benchmarks.run_length_stats_csr(matrix)
            sparsity = matrix._nnz() / np.prod(matrix.shape)

            avg.append(avg_rl / matrix.shape[1])
            max.append(max_rl / matrix.shape[1])
            sparsities.append(sparsity)

        df = pd.DataFrame({'avg': avg, 'max': max, 'sparsity': sparsities})
        df["collection"] = collection
        return df

    return pd.concat([
        _compute(ss_loader, "SuiteSparse"),
        _compute(dlmc_loader, "DLMC")
    ])


@cached_return(refresh_cache=True)
def compute_fopd_histograms():
    def _compute(loader):
        max_consecutive = 32
        weight_vector = torch.arange(0, max_consecutive+1, dtype=torch.float)

        hist_i0_f_agg = torch.zeros(max_consecutive+1)
        hist_i0_g_agg = torch.zeros(max_consecutive+1)
        hist_i0_h_agg = torch.zeros(max_consecutive+1)

        hist_i1_f_agg = torch.zeros(max_consecutive+1)
        hist_i1_g_agg = torch.zeros(max_consecutive+1)
        hist_i1_h_agg = torch.zeros(max_consecutive+1)

        for matrix, path in loader:
            print(path)

            trace = spmx_trace_gen(matrix)
            fopds_i0 = compute_consecutive_fopds(trace, dim=0)
            fopds_i1 = compute_consecutive_fopds(trace, dim=1)

            hist_i0_f = hist_mine_consecutive_fopds(fopds_i0, access_function=0, max_consecutive=max_consecutive)
            hist_i0_g = hist_mine_consecutive_fopds(fopds_i0, access_function=1, max_consecutive=max_consecutive)
            hist_i0_h = hist_mine_consecutive_fopds(fopds_i0, access_function=2, max_consecutive=max_consecutive)

            hist_i1_f = hist_mine_consecutive_fopds(fopds_i1, access_function=0, max_consecutive=max_consecutive)
            hist_i1_g = hist_mine_consecutive_fopds(fopds_i1, access_function=1, max_consecutive=max_consecutive)
            hist_i1_h = hist_mine_consecutive_fopds(fopds_i1, access_function=2, max_consecutive=max_consecutive)

            def weight_and_normalize(x):
                x = x * weight_vector
                return x / torch.sum(x)

            hist_i0_f_agg += weight_and_normalize(hist_i0_f)
            hist_i0_g_agg += weight_and_normalize(hist_i0_g)
            hist_i0_h_agg += weight_and_normalize(hist_i0_h)

            hist_i1_f_agg += weight_and_normalize(hist_i1_f)
            hist_i1_g_agg += weight_and_normalize(hist_i1_g)
            hist_i1_h_agg += weight_and_normalize(hist_i1_h)

        def normalize(x):
            return x / torch.sum(x)

        return {
            "i0": {
                "f": normalize(hist_i0_f_agg),
                "g": normalize(hist_i0_g_agg),
                "h": normalize(hist_i0_h_agg)
            },
            "i1": {
                "f": normalize(hist_i1_f_agg),
                "g": normalize(hist_i1_g_agg),
                "h": normalize(hist_i1_h_agg)
            }
        }

    return {
        "SuiteSparse": _compute(ss_loader),
        "DLMC": _compute(dlmc_loader)
    }


histograms = compute_fopd_histograms()
records = [(collection, induction_var, access_function, x, y.item())
           for collection, v0 in histograms.items()
           for induction_var, v1 in v0.items()
           for access_function, v2 in v1.items()
           for x, y in enumerate(v2)]

df = pd.DataFrame.from_records(
    records, columns=['Collection', 'Induction Variable', 'Access Function', 'Bin', 'Value'])


def plot_histogram(induction_variable, access_function):
    df_ = df[(df['Induction Variable'] == induction_variable) & (df['Access Function'] == access_function)]
    sns.barplot(x='Bin', y='Value', hue='Collection', data=df_)
    plt.gcf().set_size_inches(14, 6)
    plt.ylabel('Density')
    plt.xlabel('Strided Pattern Length')
    plt.title(f'Strided Histogram for Induction Variable {induction_variable}, Access Function {access_function}')
    plt.savefig(PLOT_DIR + f'{induction_variable}_{access_function}_hist.png')
    plt.gcf().clear()


plot_histogram('i0', 'f')
plot_histogram('i0', 'g')
plot_histogram('i0', 'h')

plot_histogram('i1', 'f')
plot_histogram('i1', 'g')
plot_histogram('i1', 'h')


# df = compute_runlengths()
# sns.histplot(data=df, x="avg", hue="collection", stat="probability", common_norm=False)
# plt.show(block=True)
