import multiprocessing
import pandas as pd
import glob

from multiprocessing import Process
from artifact.utils import *
from tools.paper.plotting.plot_utils import filter

def get_df(bcols, thread_count):
    tmp_files = glob.glob(f"/tmp/double/figure_double*.bcols{bcols}.threads{thread_count}")
    print(tmp_files)
    dfs = []
    for file in tmp_files:
        dfs.append(pd.read_csv(file))
    sp_df = pd.concat(dfs)

    tmp_files = glob.glob(f"/tmp/double/psc*.bcols{bcols}.threads{thread_count}")
    print(tmp_files)
    dfs = []
    for file in tmp_files:
        dfs.append(pd.read_csv(file))
    psc_df = pd.concat(dfs)

    # Manual horizontal merge
    for val in ["gflops/s", "time median", "correct", "required_storage"]:
        sp_df[f"{val}|PSC"] = psc_df[f"{val}|PSC"]
    return sp_df


def thread_list():
    tmp_files = glob.glob(f"/tmp/double/*.bcols*.threads*")
    thread_list = set()
    for file in tmp_files:
        thread_list.add(int(file.split("threads")[-1]))
    return list(thread_list)

def bcols_list():
    tmp_files = glob.glob(f"/tmp/double/*.bcols*.threads*")
    thread_list = set()
    for file in tmp_files:
        thread_list.add(int(file.split("bcols")[-1].split(".")[0]))
    return list(thread_list)

def post_process(df):
    nano_methods = df['name'].str.contains(r'M[0-9]N[0-9]', regex=True)
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
    df["matrixId"] = df["sparsityFolder"] + "|" + df["model"] + "|" + df["pruningMethod"] + "|" + df["matrixName"]
    df["sparsityFolder"] = round(df["sparsityFolder"].astype(float), 2)

    return df

def post_process_psc(df):
    df = pd.melt(df, id_vars=['Matrix', 'nRows', 'nCols', 'NNZ'], 
                     value_vars=['bCol=32', 'bCol=128', 'bCol=256', 'bCol=512'], 
                     var_name='bcols', 
                     value_name='time median').reset_index()
    print(df)

    # Renaming
    df["matrixPath"] = df["Matrix"]
    df["m"] = df["nRows"]
    df["k"] = df["nCols"]
    df["n"] = df["bcols"].str.split('=').str[-1].astype(int)
    df["nnz"] = df["NNZ"]
    df["numThreads"] = 20
    df["correct"] = "correct"
    df["required_storage"] = -1

    # Rescale 
    df["time median"] = df["time median"] * 1e6
    print(df['time median'])

    df['name'] = "PSC"
    df["sparsity"] = round(1 - df["nnz"] / (df["m"] * df["k"]), 2)
    df["sparsity_raw"] = 1 - df["nnz"] / (df["m"] * df["k"])
    df["pruningMethod"] = df["matrixPath"].str.split("/").str[-3]
    df["model"] = df["matrixPath"].str.split("/").str[-4]
    df["sparsityFolder"] = df["matrixPath"].str.split("/").str[-2]
    df["matrixName"] = df["matrixPath"].str.split("/").str[-1].str.replace(".mtx", "", regex=True).str.replace(".smtx", "", regex=True)
    df["matrixId"] = df["sparsityFolder"] + "|" + df["model"] + "|" + df["pruningMethod"] + "|" + df["matrixName"]
    df["sparsityFolder"] = round(df["sparsityFolder"].astype(float), 2)

    return df

def pivot(df, bcols, num_threads):
    df = filter(df, n=bcols, numThreads=num_threads)
    df = df.reset_index(drop=True)

    sparsity_buckets = pd.IntervalIndex.from_tuples([(0.0, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)])
    df["sparsity_buckets"] = pd.cut(df["sparsity"], sparsity_buckets)
    df["gflops"] = (2 * df["n"] * df["nnz"]) / 1e9
    df["gflops/s"] = (df["gflops"] / (df["time median"]/1e6))
    df["Method"] = df["name"]

    dfw = pd.pivot(df, index=["matrixId", "m", "k", "nnz", "n", "sparsity", "sparsityFolder",  "model", "pruningMethod", "matrixName", "gflops"], 
        columns=["Method"],
        values=["gflops/s", "time median", "correct", "required_storage"])

    dfw.index.names = ['Matrix', "Rows", "Cols", "NNZ", "Bcols", "sparsity", "pruningModelTargetSparsity",  "model", "pruningMethod",  "matrixName", "gflops"]
    dfw.columns = ['|'.join(col) for col in dfw.columns.values]
    dfw = dfw.reset_index()
    return dfw

def gen_post_processed_files():
    files = glob.glob(RESULTS_DIR + "/figure_double_results*.csv")

    os.makedirs('/tmp/double/', exist_ok=True)
    def pivot_process(file):
        df = pd.read_csv(file)
        df = post_process(df)
        for thread_count in df["numThreads"].unique():
            for bcols in df["n"].unique():
                dfw = pivot(df, bcols, thread_count)
                gend_file = f'/tmp/double/{file.split("/")[-1]}.bcols{bcols}.threads{thread_count}'
                dfw.to_csv(gend_file)


    processes = [Process(target=pivot_process, args=(file,)) for file in files]
    for process in processes:
        process.start()

    for process in processes:
        process.join()

    df = pd.read_csv(RESULTS_DIR + "/psc_4_bcols.csv")
    df = post_process_psc(df)
    for thread_count in df["numThreads"].unique():
        for bcols in df["n"].unique():
            print(df["time median"], "Index===>", df["time median"].index)
            dfw = pivot(df, bcols, thread_count)
            gend_file = f'/tmp/double/psc.csv.bcols{bcols}.threads{thread_count}'
            dfw.to_csv(gend_file)
            print(dfw["time median|PSC"], "Index===>", dfw["time median|PSC"].index)



if __name__ == "__main__":
    gen_post_processed_files()