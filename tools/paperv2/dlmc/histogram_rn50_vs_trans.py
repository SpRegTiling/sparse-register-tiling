import seaborn as sns
import glob
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tools.paper.plotting.plot_utils import plot_save
from matplotlib import rc, rcParams
from tools.paper.plotting.plot_utils import *
from scipy.stats import gmean
from numpy import mean

from brokenaxes import brokenaxes
from tools.paperv2.dlmc.utils import *

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 15})
plt.rcParams["figure.figsize"] = (6, 4)
plt.rcParams['axes.xmargin'] = 0
plt.rcParams['axes.ymargin'] = 0

df = read_cache("cascade", "all", bcols=128, threads=20)

# rn50 = df[df["model"] == "rn50"]["m"].tolist()
# trans = df[df["model"] == "transformer"]["m"].tolist()

#print(df["Rows"].max())

plt.figure(figsize=(4, 5.5))
bax = brokenaxes(xlims=((0, 2250), (33200, 33400)), hspace=.02)

def plot(model, color):
    dff = df[df["model"] == model]
    print(dff["Rows"].max())
    dfg = dff.groupby(['Rows','Cols']).size().reset_index().rename(columns={0:'count'})
    bax.scatter('Rows', 'Cols', 
             s='count',
             c=color,
             alpha=0.4, 
             data=dfg)
    
plot("transformer", "navy")
plot("rn50", "maroon")

for ax in bax.axs:
    ax.set_ylim(0, 5100)
bax.standardize_ticks(512, 512)
bax.set_xlabel("Rows", labelpad=20)
bax.set_ylabel("Columns", labelpad=38)
bax.set_title("DLMC Matrix Shapes")
legend = bax.legend(labels=["Transformer", "Resnet50"])

for handle in legend.legend_handles:
    handle._sizes = [30]

savefig(f"/figure_rn50_trans.pdf")

