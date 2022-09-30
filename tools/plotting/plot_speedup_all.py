import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
import sys; sys.path.insert(0,f'{SCRIPT_DIR}/../')
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from pandas.api.types import is_string_dtype
import altair as alt
import altair_saver
from collections import defaultdict
from tools.plotting.color import divergent_color_scheme
from scipy.stats.mstats import gmean
import post_process


def create_chart_grid_and_save(charts, row_width):
    charts_merged = None
    charts_row = None

    col = 0
    for i in range(0, len(charts)):
        col = i % row_width
        if col == 0:
            charts_merged = charts_row if charts_merged is None else charts_merged & charts_row
            charts_row = None

        charts_row = charts[i] if charts_row is None else charts_row | charts[i]

    if col:
        charts_merged = charts_row if charts_merged is None else charts_merged & charts_row
    return charts_merged


alt.data_transformers.enable('default', max_rows=1000000)
unique_config_columns = ["name", "n", "m_tile", "k_tile", "n_tile"]

PLOT_GFLOPS = True
PLOT_SPEEDUP = True
color_scheme = 'purpleblue'

cluster_info = {
    "graham": ("~/graham/", 256),
    "niagara": ("~/niagara/", 512),
}

cluster = "graham"
cluster_dir, vec_width = cluster_info[cluster]

files = [
    f"results/sept_5/all_dlmc_part1_{vec_width}_all_threads_large_bcols_32.csv",
    f"results/sept_5/all_dlmc_part1_{vec_width}_all_threads_small_bcols_32.csv",
    f"results/sept_5/all_dlmc_part2_{vec_width}_all_threads_large_bcols_32.csv",
    f"results/sept_5/all_dlmc_part2_{vec_width}_all_threads_small_bcols_32.csv",
    f"results/sept_5/all_dlmc_part3_{vec_width}_all_threads_large_bcols_32.csv",
    f"results/sept_5/all_dlmc_part3_{vec_width}_all_threads_small_bcols_32.csv",
]

# for file in files:
#     df = pd.read_csv(cluster_dir + file)
#     print(file, df["matrixPath"].nunique())


bcols_charts = defaultdict(lambda: [])
numThreads_charts = defaultdict(lambda: [])
PLOT_DIR = SCRIPT_DIR + "/../../plots/"

cache_file = PLOT_DIR + f"{cluster}_speedup_all_cache.csv"

if not os.path.exists(cache_file):
    dfs = []
    for file in files:
        df = pd.read_csv(cluster_dir + file)
        dfs.append(df)
    df = pd.concat(dfs)

    df["gflops"] = 2 * df["n"] * df["nnz"]
    df["gflops/s"] = (df["gflops"] / df["time median"]) / 1e9

    df = post_process.compute_speed_up_vs(df, 'MKL_Dense', group_by=["matrixPath", "n", "numThreads"])
    df = post_process.compute_matrix_properties(df)
    df = post_process.compute_best(df)
    df = post_process.compute_global_best(df)

    df.to_csv(cache_file)
else:
    df = pd.read_csv(cache_file)

df_geo_mean_speedup = df.groupby(["n", "numThreads", "name"]).agg({'Speed-up vs MKL_Dense': gmean}).reset_index()

b_colss = sorted(df["n"].unique())
n_threadss = df["numThreads"].unique()
methods = df["name"].unique()


def gflops_scatter(method, b_cols, n_threads, df):
    df_filtered = df[(df["name"] == method) & (df["n"] == b_cols) & (df["numThreads"] == n_threads)]
    return alt.Chart(
        df_filtered,
        title=[f'{method}',
               f'(b_cols: {b_cols}, numThreads: {n_threads}, matrices: {df["matrixPath"].nunique()})']
    ).mark_circle().encode(
        x=alt.X('gflops:Q', title='Problem size (gflops)', scale=alt.Scale(type='log')),
        y=alt.Y('gflops/s:Q', title='Effective GFLOPS/s'),
        color=alt.Color('sparsity:Q', scale=alt.Scale(scheme=color_scheme))
    )


def gflops_scatter_all_bcols(method, n_threads, max_gflopss, color, df):
    df_filtered = df[(df["name"] == method) & (df["numThreads"] == n_threads)]
    return alt.Chart(
        df_filtered,
        title=[f'{method}']
    ).mark_circle().encode(
        x=alt.X('gflops:Q', title='Problem size (gflops)', scale=alt.Scale(type='log')),
        y=alt.Y('gflops/s:Q', title='Effective GFLOPS/s', scale=alt.Scale(domain=[0, max_gflopss])),
        color=alt.Color(f'{color}:Q', scale=alt.Scale(scheme=color_scheme))
    )


def speedup_scatter_all_bcols(method, n_threads, max_speedup, color, df):
    df_filtered = df[(df["name"] == method) & (df["numThreads"] == n_threads)]
    return alt.Chart(
        df_filtered,
        title=[f'{method}']
    ).mark_circle().encode(
        x=alt.X('gflops:Q', title='Problem size (gflops)', scale=alt.Scale(type='log')),
        y=alt.Y('Speed-up vs MKL_Dense:Q', title='Speed-up Over Dense', scale=alt.Scale(domain=[0, max_speedup])),
        color=alt.Color(f'{color}:Q', scale=alt.Scale(scheme=color_scheme))
    )


def speedup_scatter_all_bcols(method, n_threads, max_speedup, color, df):
    df_filtered = df[(df["name"] == method) & (df["numThreads"] == n_threads)]
    return alt.Chart(
        df_filtered,
        title=[f'{method}']
    ).mark_circle().encode(
        x=alt.X('sparsity:Q', title='Sparsity'),
        y=alt.Y('Speed-up vs MKL_Dense:Q', title='Speed-up Over Dense', scale=alt.Scale(domain=[0, max_speedup])),
        color=alt.Color(f'gflops:Q', title='Problem size (gflops)', scale=alt.Scale(scheme=color_scheme))
    )


if PLOT_GFLOPS:
    for color in ["n", "sparsity"]:
        for n_threads in n_threadss:
            charts = []
            max_gflopss = df[df["numThreads"] == n_threads]["gflops/s"].max()

            for method in methods:
                print("Plotting GFLOPS for method:", method)
                chart = gflops_scatter_all_bcols(method, n_threads, max_gflopss, color, df)
                charts.append(chart)

            chart = create_chart_grid_and_save(charts, 4)
            chart = chart.resolve_scale(
                y='shared',
                color='shared'
            )
            chart = chart.properties(
                title=f'b_cols: {b_colss}, numThreads: {n_threads}, matrices: {df["matrixPath"].nunique()}'
            )
            altair_saver.save(
                chart, PLOT_DIR + f'gflops/{cluster}_gflops_all_bcols_color_{color}_n_threads_{n_threads}.png',
                fmt="png", scale_fator=4
            )

if PLOT_SPEEDUP:
    for color in ["n", "sparsity"]:
        for n_threads in n_threadss:
            charts = []
            max_speedup = df[df["numThreads"] == n_threads]["Speed-up vs MKL_Dense"].max()

            for method in methods:
                print("Plotting speedup for method:", method)
                chart = speedup_scatter_all_bcols(method, n_threads, max_speedup, color, df)
                charts.append(chart)

            chart = create_chart_grid_and_save(charts, 4)
            chart = chart.resolve_scale(
                y='shared',
                color='shared'
            )
            chart = chart.properties(
                title=f'b_cols: {b_colss}, numThreads: {n_threads}, matrices: {df["matrixPath"].nunique()}'
            )
            altair_saver.save(
                chart, PLOT_DIR + f'speedup/{cluster}_speedup_all_bcols_color_{color}_n_threads_{n_threads}.png',
                fmt="png", scale_fator=4
            )
