import altair as alt
import pandas as pd
import numpy as np


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


def plot(df):
    df["layer"] = df["matrixPath"].str.split("/").str[-1].str.split(".").str[0]
    layers = df["layer"].unique().tolist()
    repl = lambda m: f'_{int(m.group(0).split("_")[1]):2}_'
    df["layer"] = df["layer"].str.replace(r'_(\d)_', repl)

    df["name"] = df["name"].str.replace("SchedTest-", "")

    repl = lambda m: f'{m.group(0).split("x")[0]}xX'
    df["gname"] = df["name"].str.replace(r'\dx\d', repl)

    df["k_tile"] = df["config"].str.extract(r"k_tile:(\d)")
    df["k_tile"].fillna(100000, inplace=True)
    df["k_tiles"] = df["k"].astype(float) / df["k_tile"].astype(float)
    df["k_tiles"] = df["k_tiles"].apply(np.ceil).astype(int)

    df["gflops"] = 2 * df["n"] * df["nnz"]
    df["gflops/s"] = (df["gflops"] / df["time median"]) / 1e9

    df["mr"] = df["name"].str.extract(r"(\d)x\d")
    df["nr"] = df["name"].str.extract(r"\dx(\d)")

    df["nr"].fillna(4, inplace=True)
    df["mr"].fillna(1, inplace=True)

    names = df["name"].unique().tolist()

    line = ['MKL_Dense', 'MKL_Sparse']
    square = [x for x in names if 'packed' in x]
    scatter = list(set(names) - set(line) - set(square))

    chart = alt.Chart(filter(df, name=line)).mark_line().encode(
        x=alt.X('layer', title='Layer'),
        y=alt.Y('gflops/s', title='gflops/s'),
        color=alt.Color('name', title='Method'),
    )

    chart = chart + alt.Chart(filter(df, name=scatter)).mark_circle().encode(
        x=alt.X('layer', title='Layer'),
        y=alt.Y('gflops/s', title='gflops/s'),
        color=alt.Color('name', title='Method'),
    )

    chart = chart + alt.Chart(filter(df, name=square)).mark_square().encode(
        x=alt.X('layer', title='Layer'),
        y=alt.Y('gflops/s', title='gflops/s'),
        color=alt.Color('name', title='Method'),
    )

    return chart


df = pd.read_csv('/home/lwilkinson/niagara/mobilenet_70.csv')
chart = plot(df).properties(title='Scalar Cleanup Code')

df = pd.read_csv('/home/lwilkinson/niagara/mobilenet_70_3.csv')
chart = chart | plot(df).properties(title='Vectorized Cleanup Code')

df = pd.read_csv('/home/lwilkinson/niagara/mobilenet_70_1_multiple_of_16.csv')
chart = chart | plot(df).properties(title='Round up to multiple of 16')

chart.resolve_scale(y='shared').show()
