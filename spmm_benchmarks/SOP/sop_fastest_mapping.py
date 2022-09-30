import pandas as pd

df = pd.read_csv('sop_bench.csv')
df = df[df["name"] == "random (70%)"]

df_sop = df[df["method"] == "SOP4"]
print(df_sop[df_sop.time == df_sop.time.min()])
df_sop = df[df["method"] == "SOP8"]
print(df_sop[df_sop.time == df_sop.time.min()])

df = pd.read_csv('sop_bench.csv')
df = df[df["name"] == "random (80%)"]

df_sop = df[df["method"] == "SOP4"]
print(df_sop[df_sop.time == df_sop.time.min()])
df_sop = df[df["method"] == "SOP8"]
print(df_sop[df_sop.time == df_sop.time.min()])

df = pd.read_csv('sop_bench.csv')
df = df[df["name"] == "random (90%)"]

df_sop = df[df["method"] == "SOP4"]
print(df_sop[df_sop.time == df_sop.time.min()])
df_sop = df[df["method"] == "SOP8"]
print(df_sop[df_sop.time == df_sop.time.min()])