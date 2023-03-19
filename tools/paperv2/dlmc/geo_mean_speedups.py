
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

from tools.paperv2.dlmc.utils import *


def geo_mean_speedups(chipset):
    if chipset == "raspberrypi":
        mcl = [
            ("ARMCL", "lightcoral", "ARMCL (sgemm)"),
            ("XNN", "darkolivegreen", "XNNPACK (spmm, 16x1)"),
            ("Sp. Reg.", "steelblue", "Sparse Reg Tiling"),
        ]
        limits = [(0, 2.5), (0, 8)]
    else:
        mcl = [
            ("MKL_Sparse", "darkmagenta", "MKL Sparse (csr)"),
            ("MKL_Dense", "forestgreen", "MKL Dense (sgemm)"),
            ("ASpT Best", "goldenrod", "ASpT"),
            ("Sp. Reg.", "steelblue", "Sparse Reg Tiling"),
        ]
        limits = [(0, 80), (0, 800)]

    box_width = 0.15

    for threads in [1, 20] if chipset == "cascade" else [1, 4]:
        print(f"== {threads} ==")
        dfs = []
        for bcols in [32, 128, 256, 512]:
            dfs.append(read_cache(chipset, "all", bcols=bcols, threads=threads))
        df = pd.concat(dfs)

        BASLINES = {
            "cascade": ["MKL_Dense", "MKL_Sparse", "ASpT Best"],
            "raspberrypi": ["XNN", "ARMCL"]
        }[chipset]

        if chipset == "cascade":
            df.loc[~(df["correct|ASpT_increased_parallelism"] == "correct"), "time cpu median|ASpT_increased_parallelism"] = 1e16
            df.loc[~(df["correct|ASpT_increased_parallelism"] == "correct"), "gflops/s|ASpT_increased_parallelism"] = 0
        
            df["time cpu median|ASpT Best"] = df[["time cpu median|ASpT", "time cpu median|ASpT_increased_parallelism"]].min(axis=1)
            df["gflops/s|ASpT Best"] = df[["gflops/s|ASpT", "gflops/s|ASpT_increased_parallelism"]].max(axis=1)
            df["correct|ASpT Best"] = df["correct|ASpT"]


        for baseline in BASLINES:
            method = "Sp. Reg."
            df[f'Speed-up {method} vs. {baseline}'] = df[f"time cpu median|{baseline}"] / df[f"time cpu median|{method}"]
            print(f'Speed-up {method} vs. {baseline}', gmean(df[f'Speed-up {method} vs. {baseline}'].tolist()))
            
            print("Pct faster", len(df[df[f'Speed-up {method} vs. {baseline}'] > 1]) / len(df))
            dff = df[df[f'Speed-up {method} vs. {baseline}'] >= 1.5]
            print("Pct faster 1.5x in range 60% to 80%", baseline, len(dff[dff["sparsity"] <= 0.8] ) / len(dff))
            dff = df[df[f'Speed-up {method} vs. {baseline}'] >= 1.5]
            print("Pct faster 1.5x in range 70% to 95%", baseline, len(dff[dff["sparsity"] >= 0.7] ) / len(dff))
            
            
geo_mean_speedups("cascade")
print("====")
geo_mean_speedups("raspberrypi")