# create all tasks
import sys
import os
import post_process
from multiprocessing import Process
import pandas as pd
import glob
from plot_utils import *

SUBFOLDER = sys.argv[1] + '/'
CACHEFOLDER = os.path.join(RESULTS_DIR, "cache",  SUBFOLDER) + "/"

os.makedirs(CACHEFOLDER, exist_ok=True)

PER_FILE_POSTPROCESS = True
PER_PART_POSTPROCESS = True
RESTORE_GROUPS = True


##
#   Per File Postprocessing
##


def per_file_postprocess(filename):
    df = pd.read_csv(RESULTS_DIR + SUBFOLDER + filename)
    df = df[df["correct"] == "correct"]

    method_pack = "_".join(filename.split("/")[-1].split("_")[3:-3])
    df["run"] = method_pack
    df["name"] = df["name"].replace("MKL_Dense", "MKL_Dense " + method_pack)
    nano_methods = df['name'].str.contains(r'M[0-9]N[0-9]', regex=True)
    df.loc[nano_methods, 'name'] = "NANO_" + df.loc[nano_methods, 'name']
    df["is_nano"] = df['name'].str.contains(r'M[0-9]N[0-9]', regex=True)

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

    df.to_csv(CACHEFOLDER + filename.replace(".csv", "_per_file.csv"))


if PER_FILE_POSTPROCESS:
    files = [path.split('/')[-1] for path in glob.glob(RESULTS_DIR + SUBFOLDER + "/*.csv")]
    print(files)
    processes = [Process(target=per_file_postprocess, args=(file,)) for file in files]
    for process in processes:
        process.start()

    for process in processes:
        process.join()


##
#   Per Part Postprocessing
##

def per_part_postprocess(files, partname):
    print(files)
    dfs = []
    for file in files:
        dfs.append(pd.read_csv(CACHEFOLDER + file))
    df = pd.concat(dfs)

    def mkl_compute_time_vs_sparse(x):
        runs = x[x["name"] == 'MKL_Sparse']
        if not runs.empty:
            baseline = runs.iloc[0]["time median"]
            x[f'Speed-up vs MKL_Sparse'] = baseline / x["time median"]
            x[f'Speed-up vs Sparse'] = baseline / x["time median"]
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

        baseline = dense_runs.iloc[0]["time median"]
        x[f'Speed-up vs ARMCL'] = baseline / x["time median"]
        x[f'Speed-up vs Dense'] = baseline / x["time median"]
        return x

    def compute_best(x):
        x["best"] = False
        x.loc[x["time median"].idxmin(), "best"] = True
        return x

    def compute_best_nano(x):
        x["best_nano"] = False
        nanos = x[x["is_nano"] == True]
        x["num_nano"] = len(nanos)
        if not nanos.empty:
            x.loc[nanos["time median"].idxmin(), "best_nano"] = True
        return x

    print("computing for groups ...")
    speedup_vs_dense = arm_compute_time_vs_dense if "pi" in SUBFOLDER else mkl_compute_time_vs_densemulti
    speedup_vs_sparse = arm_compute_time_vs_sparse if "pi" in SUBFOLDER else mkl_compute_time_vs_sparse

    df = post_process.compute_for_group(df,
                                        [speedup_vs_dense, speedup_vs_sparse,
                                         compute_best, compute_best_nano],
                                        group_by=["matrixPath", "n", "numThreads"])
    df.to_csv(CACHEFOLDER + partname + "_per_part.csv")


if PER_PART_POSTPROCESS:
    processes = []
    for i in range(1, 6):
        files = [path.split('/')[-1] for path in glob.glob(CACHEFOLDER + f"/*dlmc_part{i}_*_per_file.csv")]
        processes.append(Process(target=per_part_postprocess, args=(files, f'part{i}')))

    for process in processes:
        process.start()

    for process in processes:
        process.join()


def filter(df, **kwargs):
    bool_index = None
    for key, value in kwargs.items():
        if isinstance(value, list):
            _bool_index = df[key].isin(value)
        else:
            _bool_index = df[key] == value
        if bool_index is None:
            bool_index = _bool_index
        else:
            bool_index = bool_index & _bool_index
    return df[bool_index]


##
#   Restore Groupings
##

if RESTORE_GROUPS:
    dfs = []
    dfs = [pd.read_csv(CACHEFOLDER + f"part{i}_per_part.csv") for i in range(1, 6)]

    df = pd.concat(dfs)

    for bcol in df["n"].unique():
        filter(df, n=bcol).to_csv(CACHEFOLDER + f"dlmc_bcols_{bcol}.csv")

    for nthread in df["numThreads"].unique():
        filter(df, numThreads=nthread).to_csv(CACHEFOLDER + f"dlmc_nthreads_{nthread}.csv")

    for bcol in df["n"].unique():
        for nthread in df["numThreads"].unique():
            filter(df, n=bcol, numThreads=nthread).to_csv(CACHEFOLDER + f"dlmc_bcols_{bcol}_nthreads_{nthread}.csv")
