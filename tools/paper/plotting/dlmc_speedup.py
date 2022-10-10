import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
import sys; sys.path.insert(0,f'{SCRIPT_DIR}/../')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.api.types import is_string_dtype

import altair as alt
import altair_saver
import os
import post_process

from plot_utils import *
from cache_utils import cached_merge_and_load, cache_df_processes
from collections import defaultdict
from tools.plotting.color import divergent_color_scheme
from scipy.stats.mstats import gmean

PLOT_DIR = SCRIPT_DIR + "/plots/"
os.makedirs(PLOT_DIR, exist_ok=True)


alt.data_transformers.enable('default', max_rows=1000000)
unique_config_columns = ["name", "n", "m_tile", "k_tile", "n_tile"]

PLOT_GFLOPS = False
PLOT_SPEEDUP = True
color_scheme = 'purpleblue'

cluster = 'niagara'
df = pd.read_csv(SCRIPT_DIR + '/.cache/dlmc_merged_postprocessed_v2.csv')

b_colss = sorted(df["n"].unique())
n_threadss = df["numThreads"].unique()
methods = df["name"].unique()
print(df["sparsityFolder"].unique())
print(df["pruningMethod"].unique())
print(methods)
print(df.columns)

methods =['MKL_Dense mkl', 'MKL_Sparse', 'MKL_Sparse_IE',
          'NANO_Best',
          'ASpT']

pruning_methods = [
    'random_pruning', 'variational_dropout'
]

fig, ax = plt.subplots(figsize=(6, 4))
p = sns.boxplot(data=df, x='sparsityFolder', y='sparsity', hue='pruningMethod', ax=ax)
p.set_title(f"Pruning method sparsity ranges")
p.set(xlabel='Sparsity', ylabel='Speedup vs MKL_Dense')
plot_save(plt, f"dlmc/boxplot_sparsity_ranges")


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

def gflops_scatter(method, b_cols, n_threads, df_filtered):
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


def gflops_scatter_all_bcols(max_gflopss, color, df_filtered):
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


def speedup_scatter_all_bcols(over, max_speedup, color, df_filtered):
    return alt.Chart(
        df_filtered,
        title=[f'{method}'],
        height=160
    ).mark_circle().encode(
        x=alt.X('sparsity:Q', title='Sparsity', scale=alt.Scale(domain=[0.5, 1.0])),
        y=alt.Y(f'{over}:Q', title=f'{over}', scale=alt.Scale(domain=[0, max_speedup])),
        color=alt.Color(f'n:Q', title='b_cols', scale=alt.Scale(scheme=color_scheme))
    )


# if PLOT_GFLOPS:
#     for color in ["n", "sparsity"]:
#         for n_threads in n_threadss:
#             charts = []
#             max_gflopss = df[df["numThreads"] == n_threads]["gflops/s"].max()
#
#             for method in methods:
#                 print("Plotting GFLOPS for method:", method)
#                 if ('NANO_Best' in method):
#                     df_filtered = df[(df["best_nano"]) & (df["numThreads"] == n_threads)]
#                 else:
#                     df_filtered = df[(df["name"] == method) & (df["numThreads"] == n_threads)]
#
#                 chart = gflops_scatter_all_bcols('Speed-up vs MKL_Sparse', max_gflopss, color, df_filtered)
#                 charts.append(chart)
#
#             chart = create_chart_grid(charts, 4)
#             chart = chart.resolve_scale(
#                 y='shared',
#                 color='shared'
#             )
#             chart = chart.properties(
#                 title=f'b_cols: {b_colss}, numThreads: {n_threads}, matrices: {df["matrixPath"].nunique()}'
#             )
#             chart_save(chart, f'gflops_{color}_all_bcols_{n_threads}_threads')

df = df[df["sparsity"] >= 0.5]

if PLOT_SPEEDUP:
    for color in ["n", "sparsity"]:
        for n_threads in n_threadss:
            df["Speed-up vs Best MKL"] = df[['Speed-up vs MKL_Dense', 'Speed-up vs MKL_Sparse']].min(axis=1)

            charts = []
            merged_chart = None
            max_speedup = 8#df[df["numThreads"] == n_threads]["Speed-up vs MKL_Dense"].max()

            for method in methods:
                print("Plotting speedup for method:", method)

                if ('NANO_Best' in method):
                    df_filtered = df[(df["best_nano"]) & (df["numThreads"] == n_threads)]
                else:
                    df_filtered = df[(df["name"] == method) & (df["numThreads"] == n_threads)]

                chart = speedup_scatter_all_bcols('Speed-up vs MKL_Sparse', max_speedup, color, df_filtered)
                charts.append(chart)

            chart = create_chart_grid(charts, 4)
            chart = chart.resolve_scale(
                y='shared',
                color='shared'
            )
            chart = chart.properties(
                title=f'Over Sparse, b_cols: {b_colss}, numThreads: {n_threads}, matrices: {df["matrixPath"].nunique()}'
            )

            merged_chart = chart
            charts = []
            max_speedup = 8


            for method in methods:
                print("Plotting speedup for method:", method)

                if ('NANO_Best' in method):
                    df_filtered = df[(df["best_nano"]) & (df["numThreads"] == n_threads)]
                else:
                    df_filtered = df[(df["name"] == method) & (df["numThreads"] == n_threads)]

                df_filtered = df_filtered[df_filtered['Speed-up vs MKL_Dense'] <= max_speedup]
                chart = speedup_scatter_all_bcols('Speed-up vs MKL_Dense', max_speedup, color, df_filtered)
                charts.append(chart)

            chart = create_chart_grid(charts, 4)
            chart = chart.resolve_scale(
                y='shared',
                color='shared'
            )
            chart = chart.properties(
                title=f'Over Dense, b_cols: {b_colss}, numThreads: {n_threads}, matrices: {df["matrixPath"].nunique()}'
            )

            merged_chart = merged_chart & chart
            charts = []
            max_speedup = 4

            for method in methods:
                print("Plotting speedup for method:", method)

                if ('NANO_Best' in method):
                    df_filtered = df[(df["best_nano"]) & (df["numThreads"] == n_threads)]
                else:
                    df_filtered = df[(df["name"] == method) & (df["numThreads"] == n_threads)]

                df_filtered = df_filtered[df_filtered['Speed-up vs Best MKL'] <= max_speedup]
                chart = speedup_scatter_all_bcols('Speed-up vs Best MKL', max_speedup, color, df_filtered)
                charts.append(chart)

            chart = create_chart_grid(charts, 4)
            chart = chart.resolve_scale(
                y='shared',
                color='shared'
            )
            chart = chart.properties(
                title=f'Over Best Dense or Sparse, b_cols: {b_colss}, numThreads: {n_threads}, matrices: {df["matrixPath"].nunique()}'
            )

            merged_chart = merged_chart & chart


    chart_save(merged_chart, f'speedup/{cluster}_speedup_all_bcols_color_{color}_n_threads_{n_threads}')



# Scaling
for n in df["n"].unique():
    df_filtered = filter(df, n=n, pruningMethod=pruning_methods, name=methods, sparsityFolder=0.8)

    fig, ax = plt.subplots(figsize=(12, 8))
    p = sns.boxplot(data=df_filtered, x='numThreads', y='scaling', hue='name', ax=ax)
    p.set_title(f"Scaling n={n}")
    p.set(xlabel='Num Threads', ylabel='Speedup vs Single Thread')
    plot_save(plt, f"dlmc/scaling/boxplot_{n}")


# Dense Speedups
for n in df["n"].unique():
    for numThreads in df["numThreads"].unique():
        df_filtered = filter(df, n=n, numThreads=numThreads, pruningMethod=pruning_methods, name=methods)
        fig, ax = plt.subplots(figsize=(12, 8))
        p = sns.boxplot(data=df_filtered, x='sparsityFolder', y='Speed-up vs MKL_Dense', hue='name', ax=ax)
        p.set_title(f"Speedup vs MKL_Dense for n={n}, numThreads={numThreads}")
        p.set(xlabel='Sparsity', ylabel='Speedup vs MKL_Dense')
        plot_save(plt, f"dlmc/vsdense/boxplot_{n}_{numThreads}")


