import torch
import numpy as np
import math
import glob
import json
import sys
import matplotlib.pyplot as plt
import numpy as np
from tools.paper.plotting.plot_utils import plot_save
from matplotlib import rc, rcParams
from tools.paper.plotting.plot_utils import *
from scipy import stats
from tabulate import tabulate
from sbench.loaders.load import load_dense
import gmpy
from collections import defaultdict

def pattern_code(vec):
    vec = vec.to(dtype=torch.int)
    pat = 0
    for idx, i in enumerate(vec):
        pat |= i.item() << idx
    return pat


def convert_panel_to_codes(A, pattern_mapping):
    _pat_codes = []
    _col_indices = []

    for i, row in enumerate(A):
        pat = pattern_code(row)
        if pat == 0: continue
        codes = pattern_mapping[pat]
        if type(codes) == int:
            _pat_codes.append(codes)
            _col_indices.append(i)
        elif type(codes) == list:
            for code in codes:
                _pat_codes.append(code)
                _col_indices.append(i)

    return torch.as_tensor(np.array(_pat_codes), dtype=torch.int), \
           torch.as_tensor(np.array(_col_indices), dtype=torch.int)


def matrix_stats(A, Mr, pattern_mapping):
    num_panels = math.ceil(A.shape[0] / Mr)
    nnz = A.sum()
    pat_exec_run = defaultdict(lambda: 0)
    pat_exec_run_pct = {}
    nnz_run = 0
    pats_total = 0

    for panel_id in range(num_panels):
        panel = A[panel_id*Mr: (panel_id + 1)*Mr, :].clone().t().contiguous()
        pat_codes, _ = convert_panel_to_codes(panel, pattern_mapping)

        for val, count in zip(*np.unique(pat_codes.numpy(), return_counts=True)):
            pat_exec_run[int(val)] += int(count)
            pats_total += count
            nnz_run += gmpy.popcount(int(val)) * count

    for key, val in pat_exec_run.items():
        pat_exec_run_pct[key] = val / pats_total

    return int(nnz.item()), nnz_run, pat_exec_run_pct, pat_exec_run, pats_total


def file_dic_todic(file_in):
    dic_pat, patterns = {}, set()
    file1 = open(file_in, 'r')
    lines = file1.readlines()
    M_r = int(lines[0])
    for line in lines[1:]:
        line = line.split(':')
        dic_pat[int(line[0])] = json.loads(line[1])
        for pat in dic_pat[int(line[0])]:
            patterns.add(pat)

    return list(patterns), dic_pat, M_r


def load_mapping_file(mapping_id):
    MAPPING_FOLDER = SCRIPT_DIR + "/../../../spmm_nano_kernels/mappings/"
    patterns, mapping, Mr = file_dic_todic(os.path.join(MAPPING_FOLDER, f'mapping_{mapping_id}.txt'))
    return patterns, mapping, Mr


mapping_map = {
    4: {"identity": "61fee", "orig": "da01e"},
    8: {"orig": "400fa", "alt": "747f9"},
}


def dump(m):
    print(m["matrixPath"], m["n"], m["sparsity"], m["sparsity_buckets"], m["sparsity_real"], m['mapping'], m['Mr'])
    mapping_id = mapping_map[int(m['Mr'])][m['mapping']]
    path = str(m["matrixPath"]).split('dlmc')[-1]
    A = load_dense(f'/sdb/datasets/dlmc/{path}')
    patterns, mapping, Mr = load_mapping_file(mapping_id)
    nnz, nnz_run, pat_exec_run_pct, pat_exec_run, pats_total = matrix_stats(A, Mr, mapping)
    filepath = SCRIPT_DIR + "/handful/" + (path.replace('.smtx', '.txt')).replace("/", "_")
    json_filepath = SCRIPT_DIR + "/handful/" + (path.replace('.smtx', '.json')).replace("/", "_")
    print(filepath)

    Mr4_pats = {i: i for i in range(2**4)}
    Mr8_pats = {i: i for i in range(2**8)}

    patterns, mapping, Mr = load_mapping_file(mapping_id)
    _, _, pat_mr_4_all_pct, pat_mr_4_all_exec_run, _ = matrix_stats(A, 4, Mr4_pats)
    _, _, pat_mr_8_all_pct, pat_mr_8_all_exec_run, _ = matrix_stats(A, 8, Mr8_pats)

    dict = {
        "mtx": path,
        "sparsity": round(100*m["sparsity_real"]),
        "rows": m["m"],
        "cols": m["k"],
        "nnz": m["nnz"],
        "bcols": m["n"],
        "Tile": str(int(Mr)) + "x" + str(int(m["Nr"])),
        "max_patterns": 2**Mr-1,
        "patterns_generated":  len(patterns),
        "patterns_run":  pats_total,
        "total_flops": nnz_run * m["n"]*2,
        "required_flops": nnz * m["n"]*2,
        "redundant_flops": (nnz_run - nnz) * m["n"]*2,
        "speedup_vs_mkl_spmm": round(m["Speed-up vs Sparse"], 2),
        "speedup_vs_mkl_sgemm": round(m["Speed-up vs Dense"], 2),
        "pat_exec_run": pat_exec_run,
        "pat_pct": pat_exec_run_pct,
        "pat_mr_4_all_exec_run": pat_mr_4_all_exec_run,
        "pat_mr_4_all_pct": pat_mr_4_all_pct,
        "pat_mr_8_all_exec_run": pat_mr_8_all_exec_run,
        "pat_mr_8_all_pct": pat_mr_8_all_pct,
    }

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    # with open(filepath, "w+") as f:
    #     f.write("mtx: " + str(path)); f.write("\n")
    #     f.write("sparsity: " + str(round(100*m["sparsity_real"]))); f.write("\n")
    #     f.write("rows: " + str(m["m"])); f.write("\n")
    #     f.write("cols: " + str(m["k"])); f.write("\n")
    #     f.write("nnz: " + str(m["nnz"])); f.write("\n")
    #     f.write("bcols: " + str(m["n"])); f.write("\n")
    #     f.write("Tile: " + str(Mr) + "x" + str(int(m["Nr"]))); f.write("\n")
    #     f.write("max_patterns: " + str(2**Mr-1)); f.write("\n")
    #     f.write("patterns_generated: " + str( len(patterns))); f.write("\n")
    #     f.write("patterns_run: " + str( pats_total)); f.write("\n")
    #     f.write("total_flops: " + str(nnz_run * m["n"]*2)); f.write("\n")
    #     f.write("required_flops: " + str(nnz * m["n"]*2)); f.write("\n")
    #     f.write("redundant_flops: " + str((nnz_run - nnz) * m["n"]*2)); f.write("\n")
    #     f.write("speedup_vs_mkl_spmm: " + str(round(m["Speed-up vs Sparse"], 2))); f.write("\n")
    #     f.write("speedup_vs_mkl_sgemm: " + str(round(m["Speed-up vs Dense"], 2))); f.write("\n")
    #     f.write("========== PATTERN EXEC COUNTS ============="); f.write("\n")
    #     for k, v in pat_exec_run.items():
    #         f.write(format(k, f'0{Mr}b'))
    #         f.write(": " + str( v)); f.write("\n")
    #     f.write("========== PATTERN PCT COUNTS ============="); f.write("\n")
    #     for k, v in pat_exec_run_pct.items():
    #         f.write(format(k, f'0{Mr}b'))
    #         f.write(": " + str(round(v*100, 2))); f.write("\n")

    with open(json_filepath, "w+") as f:
        import json

        def np_encoder(object):
            if isinstance(object, np.generic):
                return object.item()
        f.write(json.dumps(dict, default=np_encoder, indent=4))


df = load_dlmc_df("cascadelake", nthreads=16)
df = df[(df["sparsity"] <= 0.95) & (df["sparsity"] >= 0.6) & (df["n"] < 1024)]
df = df.reset_index(drop=True)
df = filter(df, best_nano=True)
df["sparsity_real"] = round(1- df["nnz"] / (df["k"] * df["m"]), 2)

sparsity_buckets = pd.IntervalIndex.from_tuples([(0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 0.95)])
df["sparsity_buckets"] = pd.cut(df["sparsity"], sparsity_buckets)

# dff = filter(df, name="NANO_M8N3_KNM_LB_orig", best_nano=True)
# m = dff.iloc[0]
# dump(m)
# m = dff.iloc[2]
# dump(m)
#
# dff = filter(df, name="NANO_M4N4_NKM_LB_SA_identity", best_nano=True)
# m = dff.iloc[2]
# dump(m)
#
# dff = filter(df, name="NANO_M4N4_NKM_LB_orig", best_nano=True)
# m = dff.iloc[2]
# dump(m)
# dff = filter(df, name="NANO_M4N4_NKM_LB_orig", model='rn50', best_nano=True)
# m = dff.iloc[6]
# dump(m)
#
# dff = filter(df, name="NANO_M8N2_KNM_LB_TLB64_SA_alt", best_nano=True)
# m = dff.iloc[6]
# dump(m)
#
# dff = filter(df, sparsity=0.6,  best_nano=True)
# m = dff.iloc[6]
# dump(m)
#
# dff = filter(df, sparsity=0.7,  best_nano=True)
# m = dff.iloc[10]
# dump(m)
#
# dff = filter(df, sparsity=0.8,  best_nano=True)
# m = dff.iloc[5]
# dump(m)

dff = filter(df, sparsity=0.6,  best_nano=True)
m = dff.iloc[2]
dump(m)

m = dff.iloc[13]
dump(m)

dff = filter(df, sparsity=0.7,  best_nano=True)
m = dff.iloc[2]
dump(m)

m = dff.iloc[13]
dump(m)

dff = filter(df, sparsity=0.8,  best_nano=True)
m = dff.iloc[2]
dump(m)

m = dff.iloc[13]
dump(m)

dff = filter(df, sparsity=0.9,  best_nano=True)
m = dff.iloc[2]
dump(m)

m = dff.iloc[13]
dump(m)