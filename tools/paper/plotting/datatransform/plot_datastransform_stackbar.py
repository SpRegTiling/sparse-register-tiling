import glob
import sys
import matplotlib.pyplot as plt
import numpy as np
from tools.paper.plotting.plot_utils import plot_save
from matplotlib import rc, rcParams
from tools.paper.plotting.plot_utils import *
from tools.paper.plotting import post_process

part = "part1"

pd.options.display.max_columns = None
pd.options.display.max_rows = None


def after_loadhook(filename, df):
    method_pack = "_".join(filename.split("/")[-1].split("_")[3:-3])
    print(filename, ("nottransformed" in filename))
    df["datatransform"] = ("nottransformed" in filename)
    df["name"] = df["name"].replace("MKL_Dense", "MKL_Dense " + method_pack)
    nano_methods = df['name'].str.contains(r'M[0-9]N[0-9]', regex=True)
    df.loc[nano_methods, 'name'] = "NANO_" + df.loc[nano_methods, 'name']
    df["is_nano"] = df['name'].str.contains(r'M[0-9]N[0-9]', regex=True)
    print(method_pack, df["is_nano"].unique(), df["datatransform"].unique())

    df["run"] = method_pack
    return df


def postprocess(df):
    df["flops"] = 2 * df["n"] * df["nnz"]
    df["gflops/s"] = (df["flops"] / (df["time median"]/1e6)) / 1e9

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

    def compute_time_vs_MKL_Sparse(x):
        runs = x[x["name"] == 'MKL_Sparse']
        if not runs.empty:
            baseline = runs.iloc[0]["time median"]
            x[f'Speed-up vs MKL_Sparse'] = baseline / x["time median"]
        return x

    def compute_time_vs_densemulti(x):
        print(x["name"])
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

    def compute_best_nano(x):
        x["best_nano"] = False
        nanos = x[x["is_nano"]]
        x["num_nano"] = len(nanos)
        if not nanos.empty:
            x.loc[nanos["time median"].idxmin(), "best_nano"] = True
        return x

    def compute_num_nano(x):
        x["num_methods"] = len(x)
        return x

    def compute_best_df_nano(x):
        x["df_best_nano"] = False
        nanos = x[(x["is_nano"]) & (x["datatransform"] == True)]
        if not nanos.empty:
            x.loc[nanos["time median"].idxmin(), "df_best_nano"] = True
        return x

    def compute_best_nodf_nano(x):
        x["nodf_best_nano"] = False
        nanos = x[(x["is_nano"]) & (x["datatransform"] == False)]
        if not nanos.empty:
            x.loc[nanos["time median"].idxmin(), "nodf_best_nano"] = True
        return x

    print("computing for groups ...")
    df = post_process.compute_for_group(df,
                                        [compute_best,
                                         compute_best_nano,
                                         compute_num_nano,
                                         compute_best_df_nano,
                                         compute_best_nodf_nano],
                                        group_by=["matrixId", "n", "numThreads"])

    # print("compute_scaling...")
    # df = post_process.compute_scaling(df)
    print("done postprocessing")

    return df


files = [
    # No Data Transform
    "/datatransform/nottransformed/dlmc_part1_AVX512_nano4_bests_part1_large_bcols_32.csv",
    "/datatransform/nottransformed/dlmc_part1_AVX512_nano4_bests_part1_small_bcols_32.csv",
    "/datatransform/nottransformed/dlmc_part1_AVX512_nano4_bests_part2_large_bcols_32.csv",
    "/datatransform/nottransformed/dlmc_part1_AVX512_nano4_bests_part2_small_bcols_32.csv",
    "/datatransform/nottransformed/dlmc_part1_AVX512_nano8_bests_part1_large_bcols_32.csv",
    "/datatransform/nottransformed/dlmc_part1_AVX512_nano8_bests_part1_small_bcols_32.csv",
    "/datatransform/nottransformed/dlmc_part1_AVX512_nano8_bests_part2_large_bcols_32.csv",
    "/datatransform/nottransformed/dlmc_part1_AVX512_nano8_bests_part2_small_bcols_32.csv",
    # Data Transform
    "/datatransform/transformed/dlmc_part1_AVX512_nano4_bests_part1_large_bcols_32.csv",
    "/datatransform/transformed/dlmc_part1_AVX512_nano4_bests_part1_small_bcols_32.csv",
    "/datatransform/transformed/dlmc_part1_AVX512_nano4_bests_part2_large_bcols_32.csv",
    "/datatransform/transformed/dlmc_part1_AVX512_nano4_bests_part2_small_bcols_32.csv",
    "/datatransform/transformed/dlmc_part1_AVX512_nano8_bests_part1_large_bcols_32.csv",
    "/datatransform/transformed/dlmc_part1_AVX512_nano8_bests_part1_small_bcols_32.csv",
    "/datatransform/transformed/dlmc_part1_AVX512_nano8_bests_part2_large_bcols_32.csv",
    "/datatransform/transformed/dlmc_part1_AVX512_nano8_bests_part2_small_bcols_32.csv",
    #
    "/datatransform/transformed/dlmc_part1_AVX512_mkl_small_bcols_32.csv",
    "/datatransform/transformed/dlmc_part1_AVX512_mkl_large_bcols_32.csv",
]


df, df_reloaded = cached_merge_and_load(files, f"dlmc_{part}_merged_df_v2",
                                        afterload_hook=after_loadhook,
                                        force_use_cache=False)
df = postprocess(df, df_reloaded)
df = filter(df, num_methods=df["num_methods"].max())

df["include"] = (df["df_best_nano"] | df["nodf_best_nano"] | df["name"].str.contains("MKL_Dense"))
df = filter(df, include=True)

df.loc[df["datatransform"] == True, "name2"] = "transformed"
df.loc[df["datatransform"] == False, "name2"] = "not-transformed"
df.loc[df["name"].str.contains("MKL_Dense"), "name2"] = "dense"

for numThreads in [1, 16, 32]:
    dftmp = filter(df, numThreads=numThreads)
    np.random.seed(0)
    paths = np.random.choice(dftmp["matrixId"].unique(), 20, replace=False)
    merged_chart = None

    for bcols in [32, 128, 256, 512, 1024]:
        dftmp2 = filter(dftmp, matrixId=list(paths), n=bcols)

        chart = alt.Chart(dftmp2).mark_bar(opacity=0.5).encode(
            x=alt.X("matrixId:N"),
            y=alt.Y("gflops/s:Q", stack=None),
            color=alt.Color("name2:N"),
        ).properties(title=f'bcols {bcols}')

        if merged_chart is None:
            merged_chart = chart
        else:
            merged_chart |= chart
    merged_chart.show()