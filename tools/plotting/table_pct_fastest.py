import pandas as pd
import argparse
import altair as alt
import altair_saver
from color import divergent_color_scheme, divergent_color_scheme_2
from post_process import post_process_demo_dataframe, post_process_demo_dataframe_papi
import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

PLOT_DIR = SCRIPT_DIR + "/../../plots/"


parser = argparse.ArgumentParser(description='Plot SpMM bench')
parser.add_argument('csv_file', help='CSV file created by SPMM_demo')

args = parser.parse_args()
df = pd.read_csv(args.csv_file)
df = post_process_demo_dataframe(df)


def compute_fastest(x):
    fastest_time = 1e12
    fastest_method = ""

    for index, row in x.iterrows():
        if row["time median"] < fastest_time:
            fastest_time = row["time median"]
            fastest_method = row["name"]
    x["fastest_method"] = fastest_method

    return x


def compute_fastest_non_MKL(x):
    fastest_time = 1e12
    fastest_method = ""

    for index, row in x.iterrows():
        if row["name"] == "MKL_Executor_Only":
            continue
        if row["time median"] < fastest_time:
            fastest_time = row["time median"]
            fastest_method = row["name"]
    x["fastest_method_non_MKL"] = fastest_method

    return x


def compute_fastest_non_MKL_256_only(x):
    fastest_time = 1e12
    fastest_method = ""

    methods_512 = ['CSR_C_1D', 'CSR_C_2D', 'CSR_C_2D_B', 'CSR_A_2D',
                   'CSR_C_1D_Tuned', 'CSR_C_2D_Tuned', 'CSR_C_2D_B_Tuned', 'CSR_A_2D_Tuned']

    for index, row in x.iterrows():
        if row["name"] == "MKL_Executor_Only" or row["name"] in methods_512:
            continue
        if row["time median"] < fastest_time:
            fastest_time = row["time median"]
            fastest_method = row["name"]
    x["fastest_method_non_MKL_256_only"] = fastest_method

    return x


df = df.groupby("matrixPath").apply(compute_fastest).reset_index(drop=True)
df = df.groupby("matrixPath").apply(compute_fastest_non_MKL).reset_index(drop=True)
df = df.groupby("matrixPath").apply(compute_fastest_non_MKL_256_only).reset_index(drop=True)

df_filtered = df[df["fastest_method_non_MKL"] == df["name"]]
print(df["fastest_method_non_MKL"].value_counts(normalize=True))

chart = alt.Chart(df_filtered, width=40).mark_circle(size=8).encode(
    x=alt.X(
        'jitter:Q',
        title=None,
        axis=alt.Axis(values=[0], ticks=True, grid=False, labels=False),
        scale=alt.Scale(),
    ),
    y=alt.Y('Speed-up vs MKL:Q'),
    color=alt.Color('fastest_method_non_MKL:N', title=["Fastest Method", "(All)"]),
    column=alt.Column('sparsity:N',
                      header=alt.Header(titleOrient='bottom', labelOrient='bottom'),
                      title="Sparsity"),
).transform_calculate(jitter='sqrt(-2*log(random()))*cos(2*PI*random())') \
    .configure_facet(spacing=0).configure_view(stroke=None)

altair_saver.save(chart, PLOT_DIR + "spmm_speed_up_fastest.pdf", fmt="pdf", scale_fator=4)


chart = alt.Chart(df_filtered, width=40).mark_circle(size=8).encode(
    x=alt.X(
        'jitter:Q',
        title=None,
        axis=alt.Axis(values=[0], ticks=True, grid=False, labels=False),
        scale=alt.Scale(),
    ),
    y=alt.Y('Speed-up vs MKL:Q'),
    color=alt.Color('fastest_method_non_MKL:N', title=["Fastest Method", "(All)"]),
    column=alt.Column('aspect ratio:N',
                      header=alt.Header(titleOrient='bottom', labelOrient='bottom'),
                      title=["Aspect Ratio", "(1 = square, short-fat < 1, tall-skinny > 1)"]),
).transform_calculate(jitter='sqrt(-2*log(random()))*cos(2*PI*random())') \
    .configure_facet(spacing=0).configure_view(stroke=None)

altair_saver.save(chart, PLOT_DIR + "spmm_speed_up_fastest_aspect.pdf", fmt="pdf", scale_fator=4)


df_filtered = df[df["fastest_method_non_MKL_256_only"] == df["name"]]
print(df["fastest_method_non_MKL_256_only"].value_counts(normalize=True))

chart = alt.Chart(df_filtered, width=40).mark_circle(size=8).encode(
    x=alt.X(
        'jitter:Q',
        title=None,
        axis=alt.Axis(values=[0], ticks=True, grid=False, labels=False),
        scale=alt.Scale(),
    ),
    y=alt.Y('Speed-up vs MKL:Q'),
    color=alt.Color('fastest_method_non_MKL_256_only:N', title=["Fastest Method", "(256 Only)"]),
    column=alt.Column('sparsity:N',
                      header=alt.Header(titleOrient='bottom', labelOrient='bottom'),
                      title="Sparsity"),
).transform_calculate(jitter='sqrt(-2*log(random()))*cos(2*PI*random())')\
    .configure_facet(spacing=0).configure_view(stroke=None)

altair_saver.save(chart, PLOT_DIR + "spmm_speed_up_fastest_256_only.pdf", fmt="pdf", scale_fator=4)

chart = alt.Chart(df_filtered, width=40).mark_circle(size=8).encode(
    x=alt.X(
        'jitter:Q',
        title=None,
        axis=alt.Axis(values=[0], ticks=True, grid=False, labels=False),
        scale=alt.Scale(),
    ),
    y=alt.Y('Speed-up vs MKL:Q'),
    color=alt.Color('fastest_method_non_MKL:N', title=["Fastest Method", "(256 Only)"]),
    column=alt.Column('aspect ratio:N',
                      header=alt.Header(titleOrient='bottom', labelOrient='bottom'),
                      title=["Aspect Ratio", "(1 = square, short-fat < 1, tall-skinny > 1)"]),
).transform_calculate(jitter='sqrt(-2*log(random()))*cos(2*PI*random())') \
    .configure_facet(spacing=0).configure_view(stroke=None)

altair_saver.save(chart, PLOT_DIR + "spmm_speed_up_fastest_aspect_256_only.pdf", fmt="pdf", scale_fator=4)