RESULTS_DIR = "/sdb/paperv2_results/"
CACHEV2_DIR = "/sdb/paperv2_results/cachev2/"
PLOTS_DIR = "/workspaces/spmm-nano-bench/plots/v2/"

import matplotlib.pyplot as plt

import os
os.makedirs(PLOTS_DIR, exist_ok=True)

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


SP_REG_COLOR = 'steelblue'
PSC_COLOR = 'peru'
MKL_SPARSE_COLOR = "darkmagenta"
MKL_DENSE_COLOR = 'forestgreen'
XNN_COLOR = 'darkolivegreen'
ASPT_COLOR = 'goldenrod'
ARMCL_COLOR = 'lightcoral'

# mcl is method, colour, label
arm_mcl = [
    ("ARMCL", ARMCL_COLOR, "ARMCL SGEMM"),
    ("XNN", XNN_COLOR, "XNNPACK"),
    ("Sp. Reg.", SP_REG_COLOR, "Sparse Register Tiling"),
]

intel_mcl = [
    ("MKL_Sparse", MKL_SPARSE_COLOR, "MKL SpMM (CSR)"),
    ("MKL_Dense", MKL_DENSE_COLOR, "MKL SGEMM"),
    ("ASpT Best", ASPT_COLOR, "ASpT"),
    ("Sp. Reg.", SP_REG_COLOR, "Sparse Register Tiling"),
]

intel_mcl_no_aspt = [
    ("MKL_Sparse", MKL_SPARSE_COLOR, "MKL SpMM (CSR)"),
    ("MKL_Dense", MKL_DENSE_COLOR, "MKL SGEMM"),
    ("Sp. Reg.", SP_REG_COLOR, "Sparse Register Tiling"),
]

intel_mcl_double = [
    ("MKL_Sparse", MKL_SPARSE_COLOR, "MKL SpMM (CSR)"),
    ("MKL_Dense", MKL_DENSE_COLOR, "MKL DGEMM"),
    ("PSC", PSC_COLOR, "LCM I/E"),
    ("Sp. Reg.", SP_REG_COLOR, "Sparse Register Tiling"),
]

def compute_aspt_best(df):
    df.loc[~(df["correct|ASpT_increased_parallelism"] == "correct"), "time cpu median|ASpT_increased_parallelism"] = 1e16
    df.loc[~(df["correct|ASpT_increased_parallelism"] == "correct"), "gflops/s|ASpT_increased_parallelism"] = 0

    df["time cpu median|ASpT Best"] = df[["time cpu median|ASpT", "time cpu median|ASpT_increased_parallelism"]].min(axis=1)
    df["gflops/s|ASpT Best"] = df[["gflops/s|ASpT", "gflops/s|ASpT_increased_parallelism"]].max(axis=1)
    df["correct|ASpT Best"] = df["correct|ASpT"]
    return df

def savefig(name):
    plt.savefig(PLOTS_DIR + name, bbox_inches = "tight")
    print("Created:", PLOTS_DIR + name)