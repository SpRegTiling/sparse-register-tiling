import glob
import sys
import matplotlib.pyplot as plt
import numpy as np
import altair as alt
from tools.paper.plotting.plot_utils import plot_save
from tools.paper.plotting.cache_utils import cached_merge_and_load, cache_df_processes
from matplotlib import rc, rcParams
from tools.paper.plotting.plot_utils import *
from tools.paper.plotting import post_process
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 30})
plt.rcParams["figure.figsize"] = (20, 7)
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0

part = "part1"

pd.options.display.max_columns = None
pd.options.display.max_rows = None


def after_loadhook(filename, df):
    method_pack = "_".join(filename.split("/")[-1].split("_")[3:-3])
    df["datatransform"] = ("nottransformed" not in filename)
    df["name"] = df["name"].replace("MKL_Dense", "MKL_Dense " + method_pack)
    nano_methods = df['name'].str.contains(r'M[0-9]N[0-9]', regex=True)
    df.loc[nano_methods, 'name'] = "NANO_" + df.loc[nano_methods, 'name']
    df["is_nano"] = df['name'].str.contains(r'M[0-9]N[0-9]', regex=True)

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
        x.loc[x['time median'] == x['time median'].min(), 'best'] = True
        return x

    def compute_best_nano(x):
        x["best_nano"] = False
        nanos = x[x["is_nano"] == True]
        x["num_nano"] = len(nanos)
        if not nanos.empty:
            x.loc[x['time median'] == nanos['time median'].min(), "best_nano"] = True
        return x

    def compute_num_nano(x):
        x["num_methods"] = len(x)
        return x

    def compute_best_df_nano(x):
        x["df_best_nano"] = False
        nanos = x[(x["is_nano"]) & (x["datatransform"] == True)]
        if not nanos.empty:
            x.loc[x['time median'] == nanos['time median'].min(), "df_best_nano"] = True
        return x

    def compute_best_nodf_nano(x):
        x["nodf_best_nano"] = False
        nanos = x[(x["is_nano"]) & (x["datatransform"] == False)]
        if not nanos.empty:
            x.loc[x['time median'] == nanos['time median'].min(), "nodf_best_nano"] = True
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
    "/sdb/paper_results/datatransform/nottransformed/dlmc_part1_AVX512_nano4_bests_part1_large_bcols_32.csv",
    "/sdb/paper_results/datatransform/nottransformed/dlmc_part1_AVX512_nano4_bests_part1_small_bcols_32.csv",
    "/sdb/paper_results/datatransform/nottransformed/dlmc_part1_AVX512_nano4_bests_part2_large_bcols_32.csv",
    "/sdb/paper_results/datatransform/nottransformed/dlmc_part1_AVX512_nano4_bests_part2_small_bcols_32.csv",
    "/sdb/paper_results/datatransform/nottransformed/dlmc_part1_AVX512_nano8_bests_part1_large_bcols_32.csv",
    "/sdb/paper_results/datatransform/nottransformed/dlmc_part1_AVX512_nano8_bests_part1_small_bcols_32.csv",
    "/sdb/paper_results/datatransform/nottransformed/dlmc_part1_AVX512_nano8_bests_part2_large_bcols_32.csv",
    "/sdb/paper_results/datatransform/nottransformed/dlmc_part1_AVX512_nano8_bests_part2_small_bcols_32.csv",
    # Data Transform
    "/sdb/paper_results/datatransform/transformed/dlmc_part1_AVX512_nano4_bests_part1_large_bcols_32.csv",
    "/sdb/paper_results/datatransform/transformed/dlmc_part1_AVX512_nano4_bests_part1_small_bcols_32.csv",
    "/sdb/paper_results/datatransform/transformed/dlmc_part1_AVX512_nano4_bests_part2_large_bcols_32.csv",
    "/sdb/paper_results/datatransform/transformed/dlmc_part1_AVX512_nano4_bests_part2_small_bcols_32.csv",
    "/sdb/paper_results/datatransform/transformed/dlmc_part1_AVX512_nano8_bests_part1_large_bcols_32.csv",
    "/sdb/paper_results/datatransform/transformed/dlmc_part1_AVX512_nano8_bests_part1_small_bcols_32.csv",
    "/sdb/paper_results/datatransform/transformed/dlmc_part1_AVX512_nano8_bests_part2_large_bcols_32.csv",
    "/sdb/paper_results/datatransform/transformed/dlmc_part1_AVX512_nano8_bests_part2_small_bcols_32.csv",
    # Dense
    "/sdb/paper_results/datatransform/transformed/dlmc_part1_AVX512_mkl_small_bcols_32.csv",
    "/sdb/paper_results/datatransform/transformed/dlmc_part1_AVX512_mkl_large_bcols_32.csv",
]

# dfs = []
# for file in files:
#     df = pd.read_csv(file)
#     if after_loadhook:
#         df = after_loadhook(file, df)
#     dfs.append(df)
# df = pd.concat(dfs)

# df = postprocess(df)
# df.to_csv("dt_postprocessed.csv")
df = pd.read_csv(f"dt_postprocessed.csv")
df = filter(df, num_methods=df["num_methods"].max())

df["include"] = (df["df_best_nano"] | df["nodf_best_nano"] | df["name"].str.contains("MKL_Dense"))
df = filter(df, include=True)

df.loc[df["datatransform"] == True, "name2"] = "transformed"
df.loc[df["datatransform"] == False, "name2"] = "not-transformed"
df.loc[df["name"].str.contains("MKL_Dense"), "name2"] = "dense"

bColsList = [32, 128, 256, 512]
numThreadsList = [1]
fig, axs = plt.subplots(len(numThreadsList), len(bColsList))
plt.locator_params(nbins=4)
dimw = 0.6
alpha = 1

adf = df[df['name2'] == 'transformed']['gflops/s'] - df[df['name2'] == 'not-transformed']['gflops/s']
adf = adf[adf.notna()]
print(adf)
a1 = (df[df['name2'] == 'transformed']['gflops/s'] - df[df['name2'] == 'not-transformed']['gflops/s']).mean()
a2 = (df[df['name2'] == 'not-transformed']['gflops/s'] - df[df['name2'] == 'dense']['gflops/s']).mean()
print(f'##### Result is: {a1/a2}')

handles, labels = [], []
for numThreads in range(len(numThreadsList)):
    dftmp = filter(df, numThreads=numThreadsList[numThreads])
    np.random.seed(0)
    paths = np.random.choice(dftmp["matrixId"].unique(), 20, replace=False)
    merged_chart = None

    for bcols in range(len(bColsList)):
        print("=======================")
        df_filtered = filter(dftmp, matrixId=list(paths), n=bColsList[bcols])

        df_filtered.sort_values(by=['sparsity'], inplace=True)
        
        # for path in df_filtered[df_filtered['name2'] == 'transformed']['matrixPath'][:-2]:
        #     print(path)

                
        for name in df_filtered[df_filtered['name2'] == 'transformed']['name'][:-2]:
            print(name)
        for name in df_filtered[df_filtered['name2'] == 'not-transformed']['name'][:-2]:
            print(name)


        # x = [val for val in range(len(df_filtered[df_filtered['name2'] == 'transformed']['sparsity']))]
        x = np.arange(len(df_filtered[df_filtered['name2'] == 'transformed']['sparsity'])-2) + 1

        axs[bcols].bar(x, df_filtered[df_filtered['name2'] == 'transformed']['gflops/s'][:-2], dimw, color='royalblue', alpha=alpha, label='unroll-and-sparse-jam + data compression')
        axs[bcols].bar(x, df_filtered[df_filtered['name2'] == 'not-transformed']['gflops/s'][:-2], dimw, color='salmon', alpha=0.8, label='unroll-and-sparse-jam')
        axs[bcols].bar(x + dimw/2, df_filtered[df_filtered['name2'] == 'dense']['gflops/s'][:-2], dimw/2, color='green', alpha=alpha, label='MKL (sgemm)')
        # axs[bcols].legend(loc='upper right')
        if bcols == 0 and numThreads == 0:
            handles.extend(axs[bcols].get_legend_handles_labels()[0])
            labels.extend(axs[bcols].get_legend_handles_labels()[1])
        # axs[bcols].set_ylabel(f'Execution Time')
        axs[bcols].spines.right.set_visible(False)
        axs[bcols].spines.top.set_visible(False)
        # axs[n, 0].grid(True)
        axs[bcols].set_xticks(x)
        # axs[bcols].set_xticklabels([str(num+1) for num in x][:-1])
        axs[bcols].set_xlabel('Matrix Instance')
        # axs[bcols].set_title(f'B Columns={bColsList[bcols]}')
        every_nth = 3
        for n, label in enumerate(axs[bcols].xaxis.get_ticklabels()):
            if n % every_nth != 0:
                label.set_visible(False)

axs[0].set_ylabel('Effective GFLOP/s')
plt.subplots_adjust(hspace=0.4, wspace=0.3)

fig.legend(handles, labels, loc='upper center', framealpha=0.3, ncol=len(handles))
# plt.show()
filepath = '/sdb/paper_plots/' + 'data_transform.pdf'
filepath = filepath.replace(".pdf", "") + ".pdf"
os.makedirs(os.path.dirname(filepath), exist_ok=True)
plt.margins(x=0)
# plt.tight_layout()
plt.tight_layout(rect=(0,0,1,0.92))
plt.savefig(filepath)