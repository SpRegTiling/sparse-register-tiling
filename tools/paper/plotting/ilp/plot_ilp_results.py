import glob
import sys
import matplotlib.pyplot as plt
import numpy as np
from tools.paper.plotting.plot_utils import plot_save
from matplotlib import rc, rcParams
from tools.paper.plotting.plot_utils import *
import seaborn as sns

files = [
    'ilp_results/sweeps/ilp/4/sop_bench_ilp_sweep_4.csv',
    'ilp_results/sweeps/ga/6/sop_bench_ilp_sweep_6.csv'
]

dfs = []
for file in files:
    df = pd.read_csv(RESULTS_DIR + file)
    df["search"] = file.split("/")[-3]
    df["M_r"] = int(file.split("/")[-2])
    df["search+M_r"] = df["search"] + " " + df["M_r"].astype(str)
    dfs.append(df)

df = pd.concat(dfs)
df["sparsity"] = df["name"].str.extract(r"(\d+)").astype(int)
print(df["sparsity"] == 70)

color_labels = df['search+M_r'].unique()
rgb_values = sns.color_palette("Set2", 4)
color_map = dict(zip(color_labels, rgb_values))


df = filter(df, sparsity=70)
print(df)
df["gflops/s"] = 2 * df["tileM"] * df["tileN"] * df["sparsity"] / df["time"]
ax = df.plot.scatter(x='num_patterns', y='gflops/s', c=df["search+M_r"].map(color_map), alpha=0.5, s=10)

print(list(df.columns))
print(df.sparsity)

plt.show(block=True)

# plt.show(block=True)
