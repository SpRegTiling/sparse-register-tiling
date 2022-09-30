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
    print("charts_merged", charts_merged)
    return charts_merged


alt.data_transformers.enable('default', max_rows=1000000)
unique_config_columns = ["name", "n", "m_tile", "k_tile", "n_tile"]

PLOT_GFLOPS = True
PLOT_SPEEDUP = True
ADD_REF_LINE = False
color_scheme = 'purpleblue'

cluster_info = {
    "graham": ("~/graham/", 256),
    "niagara": ("~/niagara/", 512),
}

cluster = "niagara"
cluster_dir, vec_width = cluster_info[cluster]

files = [
    f"results/random_sweep_{vec_width}_random_sweep_32.csv",
]

# for file in files:
#     df = pd.read_csv(cluster_dir + file)
#     print(file, df["matrixPath"].nunique())


bcols_charts = defaultdict(lambda: [])
numThreads_charts = defaultdict(lambda: [])
PLOT_DIR = SCRIPT_DIR + "/../../plots/"

cache_file = PLOT_DIR + f"{cluster}_random_sweep_cache.csv"

if not os.path.exists(cache_file) or True:
    dfs = []
    for file in files:
        df = pd.read_csv(cluster_dir + file)
        dfs.append(df)
    df = pd.concat(dfs)

    df["gflops"] = 2 * df["n"] * df["nnz"]
    df["gflops/s"] = (df["gflops"] / df["time median"]) / 1e9

    df = post_process.compute_speed_up_vs(df, 'MKL_Dense', group_by=["matrixPath", "n", "numThreads"])
    df = post_process.compute_matrix_properties(df, sparsity_round=False)
    df = post_process.compute_best(df)
    df = post_process.compute_global_best(df)

    df.to_csv(cache_file)
else:
    df = pd.read_csv(cache_file)

df_geo_mean_speedup = df.groupby(["n", "numThreads", "name"]).agg({'Speed-up vs MKL_Dense': gmean}).reset_index()

b_colss = sorted(df["n"].unique())
n_threadss = df["numThreads"].unique()
methods = df["name"].unique()
n = 256

df["gflops"] = 2 * df["n"] * df["nnz"]
df["gflops/s"] = (df["gflops"] / df["time median"]) / 1e9

df["LOADS_AND_STORES"] = df["UOPS_LOADS"] + df["UOPS_STORES"]

print(methods)

methods = {
    'CSB_CSR_32_TLB_SA': 'Tiled CSR (nr=32)',
    'CSB_CSR_64_TLB_SA': 'Tiled CSR (nr=64)',
    'N4_TLB_SA_orig': 'Nano-Kernels (nr=64, mr=4)',
    'MKL_Dense': 'MKL Dense',
    'MKL_Sparse': 'MKL Sparse'
}

# charts = []
# for n_threads in n_threadss:
#     df_filtered = df[(df["numThreads"] == n_threads) & df["name"].isin(methods.keys())]
#     chart = alt.Chart(
#         df_filtered,
#         title=[f'Random']
#     ).mark_circle().encode(
#         x=alt.X('sparsity:Q', title='Sparsity'),
#         y=alt.Y('time median:Q', title='Execution Time'),
#         color=alt.Color(f'name:N')
#     ).properties(
#         title=f'Num Threads: {n_threads}'
#     )
#
#     charts.append(chart)
#
# for n_threads in n_threadss:
#     df_filtered = df[(df["numThreads"] == n_threads) & df["name"].isin(methods)]
#     chart = alt.Chart(
#         df_filtered,
#         title=[f'Random']
#     ).mark_circle().encode(
#         x=alt.X('sparsity:Q', title='Sparsity'),
#         y=alt.Y('gflops/s:Q', title='(Throughput) GFLOPS/s'),
#         color=alt.Color(f'name:N')
#     ).properties(
#         title=f'Num Threads: {n_threads}'
#     )
#
#     charts.append(chart)

# chart = create_chart_grid_and_save(charts, 4)
# chart = chart.resolve_scale(color='shared')

# chart = chart.properties(
#     title=f'b_cols: {b_colss}, numThreads: {n_threads}, matrices: {df["matrixPath"].nunique()}'
# )
# altair_saver.save(
#     chart, PLOT_DIR + f'random/random_execution_time.png',
#     fmt="png", scale_fator=4
# )


df["density"] = 1 - df["sparsity"]
df["time_ms"] = df["time median"] * 1000
density_thresh = 1.0
density_thresh_2 = 0.3

for n in [128, 256]:
    for n_threads in n_threadss:
        charts = []

        df_filtered = df[(df["numThreads"] == n_threads) \
                         & (df["name"].isin(methods.keys())) \
                         & (df["density"] <= density_thresh) \
                         & (df["n"] == n)]

        df_filtered["name"] = df_filtered["name"].map(methods)

        chart = alt.Chart(df_filtered).mark_line().encode(
            x=alt.X('sparsity:Q', title='Sparsity'),
            y=alt.Y('time_ms:Q', title='Execution Time'),
            color=alt.Color(f'name:N')
        )
        charts.append(chart)

        # chart = alt.Chart(
        #     df_filtered,
        #     title=[f'Random']
        # ).mark_circle().encode(
        #     x=alt.X('sparsity:Q', title='Sparsity', scale=alt.Scale(domain=[0, density_thresh])),
        #     y=alt.Y('gflops/s:Q', title=['Throughput', 'Effective GFLOPS/s']),
        #     color=alt.Color(f'name:N')
        # ).properties(
        #     title=f'Num Threads: {n_threads}'
        # )
        #
        # charts.append(chart)

        chart = alt.Chart(df_filtered).transform_calculate(
            ls_ops='datum.LOADS_AND_STORES / 1000000'
        ).mark_line().encode(
            x=alt.X('sparsity:Q', title='Sparsity', scale=alt.Scale(domain=[0, density_thresh])),
            y=alt.Y('ls_ops:Q', title=['Loads + Stores', '(ops, 1e6x)']),
            color=alt.Color(f'name:N')
        )
        charts.append(chart)

        chart = alt.Chart(df_filtered).transform_calculate(
            gflops_ops='datum.SP_FLOPS_TOTAL / 1000000'
        ).mark_line().encode(
            x=alt.X('sparsity:Q', title='Sparsity', scale=alt.Scale(domain=[0, density_thresh])),
            y=alt.Y('gflops_ops:Q', title=['GFLOPs']),
            color=alt.Color(f'name:N')
        )
        charts.append(chart)

        chart = create_chart_grid_and_save(charts, 4)
        chart = chart.resolve_scale(color='shared')
        #chart = chart.properties()

        altair_saver.save(
            chart, PLOT_DIR + f'random/random_execution_time_nnz_{n_threads}_{n}.png',
            fmt="png", scale_fator=4
        )
