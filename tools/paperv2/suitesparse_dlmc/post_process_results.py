import multiprocessing
import pandas as pd
import glob

from multiprocessing import Process
from tools.paperv2.utils import *
from tools.paper.plotting.plot_utils import filter

ARCH='cascade'
OUT_DIR = RESULTS_DIR + f"cachev2/ss_dlmc/{ARCH}"

def post_process(df,  ss=False):
    nano_methods = df['name'].str.contains(r'M[0-9]N[0-9]', regex=True)
    df["orig_name"] = df['name']
    df.loc[nano_methods, 'name'] = "Sp. Reg."
    df["is_nano"] = df['name'].str.contains("Sp. Reg.")
    df["sparsity"] = round(1 - df["nnz"] / (df["m"] * df["k"]), 2)
    df["sparsity_raw"] = 1 - df["nnz"] / (df["m"] * df["k"])
    
    name_start = -4 if not ss else -1
    df["matrixName"] = df["matrixPath"].str.split("/").str[name_start:].str.join("/").str.replace(".mtx", "", regex=True).str.replace(".smtx", "", regex=True)
    
    if ss:
        df["matrixId"] = df["matrixName"]
    else:
        df["pruningMethod"] = df["matrixPath"].str.split("/").str[-3]
        df["model"] = df["matrixPath"].str.split("/").str[-4]
        df["sparsityFolder"] = df["matrixPath"].str.split("/").str[-2]
        df["matrixName"] = df["matrixPath"].str.split("/").str[-1].str.replace(".mtx", "", regex=True).str.replace(".smtx", "", regex=True)
        df["matrixPathShort"] = df["matrixPath"].str.split("/").str[-5:].str.join('/')
        df["matrixId"] = df["sparsityFolder"] + "|" + df["model"] + "|" + df["pruningMethod"] + "|" + df["matrixName"]
    df["cov"] = -1

    return df

def post_process_psc(df, ss=False):
    
    bcols = []
    for bcol in ['bCol=32', 'bCol=128', 'bCol=256', 'bCol=512']:
        if bcol in df.columns:
            bcols.append(bcol)
    
    df = pd.melt(df, id_vars=['Matrix', 'nRows', 'nCols', 'NNZ'], 
                     value_vars=bcols, 
                     var_name='bcols', 
                     value_name='time median').reset_index()
    # Renaming
    df["matrixPath"] = df["Matrix"]
    df["orig_name"] = 'PSC'
    df["m"] = df["nRows"]
    df["k"] = df["nCols"]
    df["n"] = df["bcols"].str.split('=').str[-1].astype(int)
    df["nnz"] = df["NNZ"]
    df["numThreads"] = 20
    df["correct"] = "correct"
    df["required_storage"] = -1
    df['config'] = ""

    df['name'] = "PSC"
    df["sparsity"] = round(1 - df["nnz"] / (df["m"] * df["k"]), 2)
    df["sparsity_raw"] = 1 - df["nnz"] / (df["m"] * df["k"])

    name_start = -4 if not ss else -1
    df["matrixName"] = df["matrixPath"].str.split("/").str[name_start:].str.join("/").str.replace(".mtx", "", regex=True).str.replace(".smtx", "", regex=True)
    
    if ss:
        df["matrixId"] = df["matrixName"]
    else:
        df["pruningMethod"] = df["matrixPath"].str.split("/").str[-3]
        df["model"] = df["matrixPath"].str.split("/").str[-4]
        df["sparsityFolder"] = df["matrixPath"].str.split("/").str[-2]
        df["matrixName"] = df["matrixPath"].str.split("/").str[-1].str.replace(".mtx", "", regex=True).str.replace(".smtx", "", regex=True)
        df["matrixPathShort"] = df["matrixPath"].str.split("/").str[-5:].str.join('/')
        df["matrixId"] = df["sparsityFolder"] + "|" + df["model"] + "|" + df["pruningMethod"] + "|" + df["matrixName"]

    # Rescale 
    df["time median"] = df["time median"] * 1e6
    df["time cpu median"] = df["time median"]
    df["cov"] = -1
    
    return df

def pivot(df, num_threads):
    df = filter(df, numThreads=num_threads)
    df = df.reset_index(drop=True)
    
    print("pre-pivot", len(df))

    sparsity_buckets = pd.IntervalIndex.from_tuples([(0.0, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)])
    df["sparsity_buckets"] = pd.cut(df["sparsity"], sparsity_buckets)
    df["gflops"] = (2 * df["n"] * df["nnz"]) / 1e9
    df["gflops/s"] = (df["gflops"] / (df["time median"]/1e6))
    df["Method"] = df["name"]

    dfw = pd.pivot(df, index=["matrixId", "matrixPath", "m", "k", "nnz", "n", "sparsity", "sparsity_raw", "matrixName", "gflops"], 
        columns=["Method"],
        values=["gflops/s", "time median", "time cpu median", "correct", "required_storage", "cov", "config", "orig_name"])

    print("post-pivot", len(dfw))

    dfw.index.names = ['Matrix', "MatrixPath", "Rows", "Cols", "NNZ", "Bcols", "sparsity", "sparsity_raw", "matrixName", "gflops"]
    dfw.columns = ['|'.join(col) for col in dfw.columns.values]
    dfw = dfw.reset_index()

    bool_index = None
    for method in df["Method"].unique():
        if bool_index is None:
            bool_index = (dfw[f"correct|{method}"] == "correct")
        else:
            bool_index = bool_index & (dfw[f"correct|{method}"] == "correct")
    num_bcols = dfw["Bcols"].nunique()
    dfw["all_methods_correct"] = bool_index
    def all_bcols_correct(x):
        if len(x) < num_bcols:
            x["all_methods_all_bcols_correct"] = False
            return x
        x["all_methods_all_bcols_correct"] = x["all_methods_correct"].all()
        return x

    dfw = dfw.groupby(["Matrix"], group_keys=False).apply(all_bcols_correct).reset_index(drop=True)
    return dfw[dfw["all_methods_correct"] == True]

# https://stackoverflow.com/a/5917395
def line_prepender(filename, line):
    with open(filename, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write(line.rstrip('\r\n') + '\n' + content)

def check_header(file):
    header_str = 'beta_10x,config,correct,cpufreq,iterations,k,k_tile,m,m_tile,matrixPath,n,n_tile,name,nnz,numThreads,required_storage,sparse_a,tiling_strategy,time cpu mean,time cpu median,time cpu stddev,time mean,time median,time stddev,'
    needs_header = False
    with open(file, 'r') as f:
        file_header = f.readline()
        if file_header[:6] != header_str[:6]:
            needs_header = True
    if needs_header:
        line_prepender(file, header_str)

def gen_post_processed_files():
    files = glob.glob(RESULTS_DIR + f"double/suitesparse_double*.csv")
    files += glob.glob(RESULTS_DIR + f"double/figure_double*.csv")

    os.makedirs(f'{OUT_DIR}', exist_ok=True)
    def pivot_process(file):
        #check_header(file)
        df = pd.read_csv(file)
        df = post_process(df)
        for thread_count in df["numThreads"].unique():
            dfw = pivot(df, thread_count)
            print("post-pivot-2", len(dfw))
            for bcols in df["n"].unique():
                dfwf = filter(dfw, Bcols=bcols)
                print(dfwf)
                gend_file = f'{OUT_DIR}/{file.split("/")[-1]}.bcols{bcols}.threads{thread_count}'
                print("Wrote", gend_file)
                dfwf.to_csv(gend_file)


    for file in files:
        pivot_process(file)

    # processes = [Process(target=pivot_process, args=(file,)) for file in files]
    # for process in processes:
    #     process.start()

    # for process in processes:
    #     process.join()

    df = pd.read_csv(RESULTS_DIR + f"double/psc_4_bcols_{ARCH}.csv")
    df = post_process_psc(df)
    for thread_count in df["numThreads"].unique():
        for bcols in df["n"].unique():
            df = df[df["n"]==128]
            dfw = pivot(df, thread_count)
            gend_file = f'{OUT_DIR}/psc_dlmc_{ARCH}.csv.bcols{bcols}.threads{thread_count}'
            print("Wrote", gend_file)
            dfw.to_csv(gend_file)

    df = pd.read_csv(RESULTS_DIR + f"double/psc_ss_bcols128_casc.csv")
    df = post_process_psc(df, ss=True)
    for thread_count in df["numThreads"].unique():
        for bcols in df["n"].unique():
            df = df[df["n"]==128]
            dfw = pivot(df, thread_count)
            gend_file = f'{OUT_DIR}/psc_ss_{ARCH}.csv.bcols{bcols}.threads{thread_count}'
            print("Wrote", gend_file)
            dfw.to_csv(gend_file)


if __name__ == "__main__":
    gen_post_processed_files()