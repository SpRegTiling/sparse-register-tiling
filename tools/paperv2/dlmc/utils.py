from tools.paperv2.utils import *
import pandas as pd
import os

SUBDIR = "dlmc"

def cache_file_name(chipset, identifier, **kwargs):
    dir = CACHEV2_DIR + f"/{SUBDIR}/{chipset}/"
    os.makedirs(dir, exist_ok=True)
    return dir + identifier + "_" + "_".join(f"{k}{v}" for k, v in kwargs.items()) + ".csv"
    
def read_cache(chipset, identifier, **kwargs):
    return pd.read_csv(cache_file_name(chipset, identifier, **kwargs))
    
def get_df(bcols, thread_count):
    tmp_files = glob.glob(f"/tmp/figure7_to_9_pivoted/*.bcols{bcols}.threads{thread_count}")
    dfs = []
    for file in tmp_files:
        dfs.append(pd.read_csv(file))
    return pd.concat(dfs)


def post_process(df):
    nano_methods = df['name'].str.contains(r'M[0-9]N[0-9]', regex=True) | df['name'].str.contains(r'NANO', regex=True) 
    df["orig_name"] = df['name']
    df.loc[nano_methods, 'name'] = "Sp. Reg."
    df["is_nano"] = df['name'].str.contains("Sp. Reg.")
    df["is_aspt"] = df['name'].str.contains("ASpT")
    df["sparsity"] = round(1 - df["nnz"] / (df["m"] * df["k"]), 2)
    df["sparsity_raw"] = 1 - df["nnz"] / (df["m"] * df["k"])
    df["pruningMethod"] = df["matrixPath"].str.split("/").str[-3]
    df["model"] = df["matrixPath"].str.split("/").str[-4]
    df["sparsityFolder"] = df["matrixPath"].str.split("/").str[-2]
    df["matrixName"] = df["matrixPath"].str.split("/").str[-1].str.replace(".mtx", "", regex=True).str.replace(".smtx", "", regex=True)
    df["matrixPathShort"] = df["matrixPath"].str.split("/").str[-5:].str.join('/')
    df["matrixId"] = df["sparsityFolder"] + "|" + df["model"] + "|" + df["pruningMethod"] + "|" + df["matrixName"]
    df["sparsityFolder"] = round(df["sparsityFolder"].astype(float), 2)

    return df

def pivot(df, drop_dupes=False, **kwargs):
    df = filter(df, **kwargs)
    df = df.reset_index(drop=True)

    sparsity_buckets = pd.IntervalIndex.from_tuples([(0.0, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)])
    df["sparsity_buckets"] = pd.cut(df["sparsity"], sparsity_buckets)
    df["gflops"] = (2 * df["n"] * df["nnz"]) / 1e9
    df["gflops/s"] = (df["gflops"] / (df["time median"]/1e6))
    df["Method"] = df["name"]

    index = ["matrixId", "matrixPathShort", "m", "k", "nnz", "n", "numThreads", "sparsity", "sparsity_raw",  "sparsityFolder", "model", "pruningMethod", "matrixName", "gflops"]
    columns = ["Method"]

    if drop_dupes:
        df = df.drop_duplicates(index + columns)
    
    dfw = pd.pivot(df, index=index, columns=columns,
        values=["gflops/s", "time cpu median", "time median", "correct", "required_storage", "config"])

    dfw.index.names = ['Matrix', "MatrixPath", "Rows", "Cols", "NNZ", "Bcols", "numThreads", "sparsity",  "sparsity_raw", "pruningModelTargetSparsity",  "model", "pruningMethod",  "matrixName", "gflops"]
    dfw.columns = ['|'.join(col) for col in dfw.columns.values]
    return dfw
