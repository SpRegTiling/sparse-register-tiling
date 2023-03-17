import multiprocessing
import pandas as pd
import glob
import re
import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__)) + "/"

from multiprocessing import Process
from tools.paperv2.utils import *
from tools.paperv2.suitesparse_dlmc.post_process_results import ARCH, OUT_DIR
from tools.paper.plotting.plot_utils import filter

import subprocess

DATASET_DIR = "/datasets/"
SOURCE_ROOT = SCRIPT_DIR + "../../../"
COV_BINARY = SOURCE_ROOT + "release-build/cpp_testbed/demo/tile_cov"

THREAD_COUNT=20
BCOLS=128

CACHE_DIR = RESULTS_DIR + "/cachev2/suitesparse_dlmc/"
os.makedirs(CACHE_DIR, exist_ok=True)

if __name__ == '__main__':
    tmp_files = glob.glob(f"{OUT_DIR}/suitesparse_double*.bcols{BCOLS}.threads{THREAD_COUNT}")
    dfs = []
    for file in tmp_files:
        dfs.append(pd.read_csv(file))
    df = pd.concat(dfs)

    for index, row in df.iterrows():
        if type(row['config|Sp. Reg.']) == str and "k_tile" in row['config|Sp. Reg.']:
            m_tile = int(re.search(r"m_tile:(\d+)", row['config|Sp. Reg.']).group(1))
            k_tile = int(re.search(r"k_tile:(\d+)", row['config|Sp. Reg.']).group(1))
            path = DATASET_DIR + "/suitesparse/" + row["MatrixPath"].split("/suitesparse/")[-1]
            cmd = [COV_BINARY, '-m', path, '--ti', str(m_tile), '--tk', str(k_tile)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            print(float(result.stdout.split(",")[0]))
            print(float(result.stdout.split(",")[-1]))
            assert result.stdout.split(",")[1] == result.stdout.split(",")[2], " ".join(cmd)
            df.loc[index, 'cov'] = float(result.stdout.split(",")[0])
        else:
            df.loc[index, 'cov'] = -1
    df.to_csv(CACHE_DIR + "ss_with_cov.csv")