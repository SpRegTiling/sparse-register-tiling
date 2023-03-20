import sys
import os
import tools.paper.plotting.post_process as post_process
from multiprocessing import Process
import pandas as pd
import glob
from tools.paper.plotting.plot_utils import *

##
#   Per File Postprocessing
##

def per_file_postprocess(df, chipset):
    df = df[df["correct"] == "correct"]
    nano_methods = df['name'].str.contains(r'M[0-9]N[0-9]', regex=True)
    df.loc[nano_methods, 'name'] = "NANO_" + df.loc[nano_methods, 'name']
    df["is_nano"] = df['name'].str.contains("NANO")
    df["is_aspt"] = df['name'].str.contains("ASpT")

    df["gflops"] = 2 * df["n"] * df["nnz"]
    df["gflops/s"] = (df["gflops"] / df["time median"]) / 1e9

    print("compute_matrix_properties ...")
    df = post_process.compute_matrix_properties(df)

    print("compute_pruning_method_and_model ...")
    df = post_process.compute_pruning_method_and_model(df)

    print("compute schedule ...")
    df["schedule"] = df["name"].str.extract(r'(NKM|KNM)', expand=False)
    df["mapping"] = df["name"].str.extract(r'(identity|orig|alt)', expand=False)
    df["Mr"] = df["name"].str.extract(r'M(\d)', expand=False)
    df["Nr"] = df["name"].str.extract(r'N(\d)', expand=False)

    df["Mr"] = pd.to_numeric(df['Mr'], errors='coerce').astype("Int32")
    df["Nr"] = pd.to_numeric(df['Nr'], errors='coerce').astype("Int32")

    print("compute_scaling...")
    df = post_process.compute_scaling(df)
    print("done per file postprocessing")

    def mkl_compute_time_vs_sparse(x):
        runs = x[x["name"] == 'MKL_Sparse']
        if not runs.empty:
            baseline = runs.iloc[0]["time median"]
            x[f'Speed-up vs MKL_Sparse'] = baseline / x["time median"]
            x[f'Speed-up vs Sparse'] = baseline / x["time median"]
        return x

    def compute_time_vs_aspt(x):
        runs = x[x["name"] == 'ASpT']
        if not runs.empty:
            baseline = runs.iloc[0]["time median"]
            x[f'Speed-up vs ASpT'] = baseline / x["time median"]
        return x

    def arm_compute_time_vs_sparse(x):
        runs = x[x["name"] == 'XNN']
        if not runs.empty:
            baseline = runs.iloc[0]["time median"]
            x[f'Speed-up vs XNN Sparse'] = baseline / x["time median"]
            x[f'Speed-up vs Sparse'] = baseline / x["time median"]
        return x

    def mkl_compute_time_vs_densemulti(x):
        dense_runs = x[x["name"] == f'MLK_Dense mkl']
        if dense_runs.empty:
            dense_runs = x[x["name"].str.contains("MKL_Dense")]

        baseline = dense_runs.iloc[0]["time median"]
        x[f'Speed-up vs MKL_Dense'] = baseline / x["time median"]
        x[f'Speed-up vs Dense'] = baseline / x["time median"]
        return x

    def arm_compute_time_vs_dense(x):
        dense_runs = x[x["name"] == f'ARMCL']
        if dense_runs.empty:
            dense_runs = x[x["name"].str.contains("ARMCL")]

        try:
            baseline = dense_runs.iloc[0]["time median"]
            x[f'Speed-up vs ARMCL'] = baseline / x["time median"]
            x[f'Speed-up vs Dense'] = baseline / x["time median"]
        except:
            x[f'Speed-up vs ARMCL'] = -1
            x[f'Speed-up vs Dense'] = -1
        return x

    def compute_best(x):
        x["best"] = False
        x.loc[x['time median'] == x['time median'].min(), 'best'] = True
        return x

    def compute_best_nano(x):
        x["best_nano"] = False
        nanos = x[x["is_nano"] == True]
        x["num_nano"] = len(nanos)
        if not nanos.empty:
            x.loc[x['time median'] == nanos['time median'].min(), "best_nano"] = True
        return x

    def compute_best_aspt(x):
        x["best_aspt"] = False
        nanos = x[x["is_aspt"] == True]
        x["num_aspt"] = len(nanos)
        if not nanos.empty:
            x.loc[x['time median'] == nanos['time median'].min(), "best_aspt"] = True
        return x

    print("computing for groups ...")
    speedup_vs_dense = arm_compute_time_vs_dense if "pi" in chipset else mkl_compute_time_vs_densemulti
    speedup_vs_sparse = arm_compute_time_vs_sparse if "pi" in chipset else mkl_compute_time_vs_sparse

    df = post_process.compute_for_group(df,
                                        [speedup_vs_dense, speedup_vs_sparse, compute_time_vs_aspt,
                                         compute_best, compute_best_nano, compute_best_aspt],
                                        group_by=["matrixId", "n", "numThreads"])
    return df