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

alt.data_transformers.enable('default', max_rows=1000000)

unique_config_columns = ["name", "n", "m_tile", "k_tile", "n_tile"]

files = [
    # "rn50_magnitude_70_hybrid_ie_results.csv",
    # "rn50_magnitude_80_hybrid_ie_results.csv",
    # "rn50_variational_70_80_hybrid_ie_results.csv",
    # "transformer_magnitude_70_hybrid_ie_results.csv",
    # "transformer_magnitude_80_hybrid_ie_results.csv",
    # "transformer_variational_70_80_hybrid_ie_results.csv",
    # "transformer_random_70_hybrid_ie_results.csv",
    # "transformer_random_80_hybrid_ie_results.csv",
    "transformer_magnitude_80_hybrid_ie_packed_results_parallel.csv",
]


def parse_file_name(filename):
    model, pruning, sparsity_start, sparsity_end, *rest = filename.split("_")

    sparsity = sparsity_start
    if sparsity_end != "hybrid":
        sparsity += " - " + sparsity_end
    sparsity += "%"

    return model, pruning, sparsity


bcols_charts = defaultdict(lambda: [])
numThreads_charts = defaultdict(lambda: [])
PLOT_DIR = SCRIPT_DIR + "/../../plots/"


def create_chart_grid(name, charts, row_width, title=None):
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

    if title is not None:
        charts_merged = charts_merged.properties(title=title)

    #charts_merged.show()
    altair_saver.save(charts_merged, PLOT_DIR + f'{name}.png', fmt="png", scale_fator=4)


for file in files:
    model, pruning, sparsity = parse_file_name(file)
    sparsity_filestr = sparsity.replace(" - ", "_").replace("%","")
    df = pd.read_csv("~/niagara/" + file)

    df = post_process.compute_speed_up_vs(df, 'MKL_Dense', group_by=["matrixPath", "n", "numThreads"])
    df = post_process.compute_matrix_properties(df)
    df = post_process.compute_best(df)
    df = post_process.compute_global_best(df)

    avg_speedup = post_process.compute_avg_speedup_per_config(df, speed_up_vs='MKL_Dense', group_by=unique_config_columns)
    df["name"] = df["name"].str.replace("TILED_ROW_BASED_", "")

    df_geo_mean_speedup = df.groupby(["n", "numThreads", "name"]).agg({'Speed-up vs MKL_Dense': gmean}).reset_index()
    print(df["sparsity"])
    print(df["matrixPath"][0])
    print(df["correct"].value_counts())

    def heatmap_chart(name, filter=False):
        df_to_chart = df_geo_mean_speedup[df_geo_mean_speedup['name'] == name]

        if filter:
            df_to_chart = df_to_chart[df_to_chart['n'] != 64]

        return alt.Chart(
            df_to_chart,
            width=150,
            height=150,
            title=name
        ).mark_rect().encode(
            x='n:O',
            y='numThreads:O',
            color=alt.Color('Speed-up vs MKL_Dense:Q', scale=alt.Scale(scheme='redyellowgreen', domainMid=1))
        )


    def speedup_line_vs_threads(n):
        df_to_chart = df_geo_mean_speedup[(df_geo_mean_speedup["n"] == n)]
        return alt.Chart(
            df_to_chart,
            title=[f'{model}, {pruning}, {sparsity}', f'(bcols: {n}, matrices: {df["matrixPath"].nunique()})']
        ).mark_line(point=True).encode(
            x=alt.X('numThreads:Q'),
            y=alt.Y('Speed-up vs MKL_Dense:Q'),
            color=alt.Color('name:N')
        )

    def speedup_line_vs_bcols(n):
        df_to_chart = df_geo_mean_speedup[(df_geo_mean_speedup["numThreads"] == n)]
        return alt.Chart(
            df_to_chart,
            title=name
        ).mark_line(point=True).encode(
            x=alt.X('n:Q', title='bCols'),
            y=alt.Y('Speed-up vs MKL_Dense:Q'),
            color=alt.Color('name:N')
        )

    heatmaps = []
    for name in df['name'].unique():
        if name == "MKL_Dense": continue

        chart = heatmap_chart(name)
        heatmaps.append(chart)

    create_chart_grid(f'heatmap_{model}_{pruning}_{sparsity_filestr}', heatmaps, 3,
                      title=[f'{model}, {pruning}, {sparsity}', f'(matrices: {df["matrixPath"].nunique()})'])

    heatmaps = []
    for name in df['name'].unique():
        if name == "MKL_Dense": continue

        chart = heatmap_chart(name, filter=True)
        heatmaps.append(chart)

    create_chart_grid(f'heatmap_{model}_{pruning}_{sparsity_filestr}_filtered', heatmaps, 3,
                      title=[f'{model}, {pruning}, {sparsity}', f'(matrices: {df["matrixPath"].nunique()})'])

    merged_charts = None
    for bCols in df["n"].unique():
        chart = speedup_line_vs_threads(bCols)
        bcols_charts[bCols].append(chart)
        merged_charts = chart if merged_charts is None else merged_charts | chart

    altair_saver.save(merged_charts, PLOT_DIR + f'num_threads_packed_{model}_{pruning}_{sparsity_filestr}.png',
                      fmt="png", scale_fator=4)

    merged_charts = None
    for numThreads in df["numThreads"].unique():
        chart = speedup_line_vs_bcols(numThreads)
        numThreads_charts[numThreads].append(chart)
        merged_charts = chart if merged_charts is None else merged_charts | chart

    altair_saver.save(merged_charts, PLOT_DIR + f'bcols_{model}_{pruning}_{sparsity_filestr}.png',
                      fmt="png", scale_fator=4)


print(bcols_charts)

for bCols in [64, 128, 256, 1024, 2048]:
    create_chart_grid(f'global_packed_threads_bcols_{bCols}', bcols_charts[bCols], 2)

for numThreads in [1]:
    create_chart_grid(f'global_packed_bcols_num_threads_{numThreads}', numThreads_charts[numThreads], 3)
