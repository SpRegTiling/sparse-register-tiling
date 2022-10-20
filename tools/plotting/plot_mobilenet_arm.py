import pandas as pd
import altair as alt
import altair_saver
import numpy as np

df = pd.read_csv('mobilenet_70.csv')
print(df.columns)

df["layer"] = df["matrixPath"].str.split("/").str[-1].str.split(".").str[0]
layers = df["layer"].unique().tolist()

repl = lambda m: f'{m.group(0).split("x")[0]}xX'
df["gname"] = df["name"].str.replace(r'\dx\d', repl)

df["k_tile"] = df["config"].str.extract(r"k_tile:(\d)")
df["k_tile"].fillna(100000, inplace=True)
df["k_tiles"] = df["k"].astype(float) / df["k_tile"].astype(float)
df["k_tiles"] = df["k_tiles"].apply(np.ceil).astype(int)


df["mr"] = df["name"].str.extract(r"(\d)x\d")
df["nr"] = df["name"].str.extract(r"\dx(\d)")

df["nr"].fillna(4, inplace=True)
df["mr"].fillna(1, inplace=True)

def nr_norm(x):
        print(x)
    x["nr_norm"] = (x["nr"].astype(int)  x["nr"].astype(int).max()) / (2*x["nr"].astype(int).max())
    return x

def mr_norm(x):
        x["mr_norm"] = (x["mr"].astype(int)  x["mr"].astype(int).max()) / (2*x["mr"].astype(int).max())
    return x

df = df.groupby(["mr"], group_keys=False).apply(nr_norm).reset_index(drop=True)
df = df.groupby(["nr"], group_keys=False).apply(mr_norm).reset_index(drop=True)


df["split"] = df["name"].str.extract(r"split_([A-Z])")
df["lb"] = df["name"].str.extract(r"_(load_balanced)")
df["L1_HIT_RATE"] = (df["L1_ACCESS_TOTAL"] - df["L1_MISS_TOTAL"]) / df["L1_ACCESS_TOTAL"]
df["BRANCH_TOTAL"] = (df["BRANCH_MISPRED"]  df["BRANCH_MISPRED"])

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


df_t1 = filter(df, numThreads=1)
df_t4 = filter(df, numThreads=4)


print(df.columns)
merged_chart = None
for layer in layers:
        df_layer = filter(df_t1, layer=layer)
    chart = alt.Chart(df_layer, width=200, height=200).mark_circle().encode(
            x=alt.X('L1_HIT_RATE:Q', scale=alt.Scale(domain=[0.5, 1])),
                       y=alt.Y('time median:Q'),
                                  opacity=alt.Opacity('nr_norm:Q'),
                                                   color=alt.Color('gname:N'),
                                                              )
    layer_chart = chart

    chart = alt.Chart(df_layer, width=200, height=200).mark_circle().encode(
            x=alt.X('LOADS:Q'),
                       y=alt.Y('time median:Q'),
                                  opacity=alt.Opacity('nr_norm:Q'),
                                                   color=alt.Color('gname:N'),
                                                              )
    layer_chart = layer_chart | chart

    chart = alt.Chart(df_layer, width=200, height=200).mark_circle().encode(
            x=alt.X('STORES:Q'),
                       y=alt.Y('time median:Q'),
                                  opacity=alt.Opacity('nr_norm:Q'),
                                                   color=alt.Color('gname:N'),
                                                              )
    layer_chart = layer_chart | chart

    chart = alt.Chart(df_layer, width=200, height=200).mark_circle().encode(
            x=alt.X('BRANCH_MISPRED:Q'),
                       y=alt.Y('time median:Q'),
                                  opacity=alt.Opacity('nr_norm:Q'),
                                                   color=alt.Color('gname:N'),
                                                              )
    layer_chart = layer_chart | chart

    chart = alt.Chart(df_layer, width=200, height=200).mark_circle().encode(
            x=alt.X('UNALIGNED_ACCESS:Q'),
                       y=alt.Y('time median:Q'),
                                  opacity=alt.Opacity('nr_norm:Q'),
                                                   color=alt.Color('gname:N'),
                                                              )
    layer_chart = layer_chart | chart

    chart = alt.Chart(df_layer, width=200, height=200).mark_circle().encode(
            x=alt.X('L1I_MISS_TOTAL:Q'),
                       y=alt.Y('time median:Q'),
                                  opacity=alt.Opacity('nr_norm:Q'),
                                                   color=alt.Color('gname:N'),
                                                              )
    layer_chart = layer_chart | chart

    chart = alt.Chart(df_layer, width=200, height=200).mark_circle().encode(
            x=alt.X('L1I_TLB:Q'),
                       y=alt.Y('time median:Q'),
                                  opacity=alt.Opacity('nr_norm:Q'),
                                                   color=alt.Color('gname:N'),
                                                              )
    layer_chart = layer_chart | chart

    chart = alt.Chart(df_layer, width=200, height=200).mark_circle().encode(
            x=alt.X('BRANCH_TOTAL:Q'),
                       y=alt.Y('time median:Q'),
                                  opacity=alt.Opacity('nr_norm:Q'),
                                                   color=alt.Color('gname:N'),
                                                              )
    layer_chart = layer_chart | chart

    chart = alt.Chart(df_layer, width=200, height=200).mark_circle().encode(
            x=alt.X('k_tiles:Q'),
                       y=alt.Y('STORES:Q'),
                                  opacity=alt.Opacity('nr_norm:Q'),
                                                   color=alt.Color('gname:N'),
                                                              )
    layer_chart = layer_chart | chart


    if merged_chart:
            merged_chart = merged_chart & layer_chart
    else:
        merged_chart = layer_chart

merged_chart = merged_chart.resolve_scale(color='independent')
altair_saver.save(merged_chart, 'mobilenet_png.png', fmt="png", scale_fator=4)