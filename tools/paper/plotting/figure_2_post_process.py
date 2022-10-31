import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
import sys; sys.path.insert(0,f'{SCRIPT_DIR}/../')
import modin.pandas as pd
import glob

import altair as alt
import altair_saver
import os
import post_process

from plot_utils import *
from cache_utils import cached_merge_and_load, cache_df_processes
from collections import defaultdict
from tools.plotting.color import divergent_color_scheme
from scipy.stats.mstats import gmean


alt.data_transformers.enable('default', max_rows=1000000)



cluster_info = {
    "graham": ("/home/lwilkinson/graham", "AVX2", 32),
    "niagara": ("/home/lwilkinson/niagara/2", "AVX512", 32),
}

cluster = "niagara"
cluster_dir, arch, threads = cluster_info[cluster]
subdir = ""


def after_loadhook(filename, df):
    method_pack = "_".join(filename.split("/")[-1].split("_")[6:-3])
    df["name"] = df["name"].replace("MKL_Dense", "MKL_Dense " + method_pack)
    nano_methods = df['name'].str.contains(r'M[0-9]N[0-9]', regex=True)
    df.loc[nano_methods, 'name'] = "NANO_" + df.loc[nano_methods, 'name']

    df["run"] = method_pack
    return df


files = glob.glob(f"{cluster_dir}/results/{subdir}/random_sweep_select_{arch}_*.csv")
print(files)
files = [f for f in files if "sota" not in f]
df, df_reloaded = cached_merge_and_load(files, "figure_2_merged",
                                        afterload_hook=after_loadhook,
                                        force_use_cache=False)


@cache_df_processes("figure_2_merged_postprocessed")
def postprocess(df):
    df["gflops"] = 2 * df["n"] * df["nnz"]
    df["gflops/s"] = (df["gflops"] / df["time median"]) / 1e9

    print("compute_matrix_properties ...")
    df = post_process.compute_matrix_properties(df)

    def compute_time_vs_MKL_Sparse(x):
        runs = x[x["name"] == 'MKL_Sparse']
        if not runs.empty:
            baseline = runs.iloc[0]["time median"]
            x[f'Speed-up vs MKL_Sparse'] = baseline / x["time median"]
        return x

    def compute_time_vs_densemulti(x):
        dense_runs = x[x["name"] == f'MLK_Dense mkl']
        if dense_runs.empty:
            dense_runs = x[x["name"].str.contains("MKL_Dense")]

        baseline = dense_runs.iloc[0]["time median"]
        x[f'Speed-up vs MKL_Dense'] = baseline / x["time median"]
        return x

    def compute_best(x):
        x["best"] = False
        x.loc[x["time median"].idxmin(), "best"] = True
        return x

    def compute_global_best(x):
        x["global_best"] = False
        x.loc[x["time median"].idxmin(), "global_best"] = True
        return x

    def compute_nano(x):
        x["best_nano"] = False
        nanos = x[x["name"].str.contains("NANO")]
        if not nanos.empty:
            x.loc[nanos["time median"].idxmin(), "best_nano"] = True
        return x

    print("computing for groups ...")
    df = post_process.compute_for_group(df,
                                        [compute_time_vs_MKL_Sparse, compute_time_vs_densemulti,
                                         compute_best, compute_global_best, compute_nano],
                                        group_by=["matrixPath", "n", "numThreads"])

    print("compute_scaling...")
    df = post_process.compute_scaling(df)
    print("done postprocessing")

    return df

df = postprocess(df, df_reloaded)
