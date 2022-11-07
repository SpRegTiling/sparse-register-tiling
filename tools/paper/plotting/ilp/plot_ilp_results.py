import glob
import sys

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from tools.paper.plotting.plot_utils import plot_save
from matplotlib import rc, rcParams
from tools.paper.plotting.plot_utils import *
import seaborn as sns


files = [
    ('ilp_results/sweep_mappings/ilp/4/ilp_sweep_4_24_128_32.csv', 'ilp4'),
    ('ilp_results/sweep_mappings/ga/4/ilp_sweep_4_24_128_32.csv', 'ga4'),
    ('ilp_results/sweep_mappings/ilp/6/ilp_sweep_6_24_128_32.csv', 'ilp6'),
    ('ilp_results/sweep_mappings/ga/6/ilp_sweep_6_24_128_32.csv', 'ga6'),
    ('ilp_results/sweep_mappings/ga/8/ilp_sweep_8_24_128_32.csv', 'ga8'),
]


dfs = []
for file, name in files:
    df = pd.read_csv(RESULTS_DIR + file)
    df["search"] = name
    df["M_r"] = int(file.split("/")[-2])
    df["search+M_r"] = df["search"] + " " + df["M_r"].astype(str)
    dfs.append(df)

df = pd.concat(dfs)
df["sparsity"] = df["name"].str.extract(r"(\d+)").astype(int)
print(df["sparsity"] == 70)

color_labels = df['search'].unique()
rgb_values = sns.color_palette("Set2", 5)
color_map = dict(zip(color_labels, rgb_values))


df = filter(df, sparsity=70)
print(df["tileM"], df["tileN"])
df["gflops/s"] = 2 * df["tileM"] * df["tileN"] * ((100-df["sparsity"]) / 100) / df["time"]
ax = df.plot.scatter(x='num_patterns', y='gflops/s', c=df["search"].map(color_map), alpha=0.5, s=10)

print(list(df.columns))
print(df.sparsity)

markers = []
for name, color in color_map.items():
    markers.append(mlines.Line2D([], [], color=color, marker='o', linestyle='None', markersize=10, label=name))

plt.legend(handles=markers)
plot_save("ilp/sweep")
