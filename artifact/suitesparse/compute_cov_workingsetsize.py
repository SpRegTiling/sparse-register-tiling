import multiprocessing
import pandas as pd
import glob
import re
import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__)) + "/"

from multiprocessing import Process
from artifact.utils import *
from tools.paper.plotting.plot_utils import filter
from artifact.suitesparse.post_process_results import ARCH, check_header

import subprocess

DATASET_DIR = "/datasets/"
SOURCE_ROOT = SCRIPT_DIR + "../../"
COV_BINARY = SOURCE_ROOT + "release-build/cpp_testbed/demo/tile_cov"

if __name__ == '__main__':
    files = glob.glob(RESULTS_DIR + f"{ARCH}/suitesparse_double*.csv")
    print(files)

    os.makedirs(f'/tmp/suitesparse/{ARCH}', exist_ok=True)
    def compute_cov_process(file):
        check_header(file)
        df = pd.read_csv(file)
        for index, row in df.iterrows():
            if type(row['config']) == str and "k_tile" in row['config']:
                m_tile = int(re.search(r"m_tile:(\d+)", row['config']).group(1))
                k_tile = int(re.search(r"k_tile:(\d+)", row['config']).group(1))
                path = DATASET_DIR + "/suitesparse/" + row["matrixPath"].split("/suitesparse/")[-1]
                cmd = [COV_BINARY, '-m', path, '--ti', str(m_tile), '--tk', str(k_tile)]
                result = subprocess.run(cmd, capture_output=True, text=True)
                print(float(result.stdout.split(",")[0]))
                print(float(result.stdout.split(",")[-1]))
                assert result.stdout.split(",")[1] == result.stdout.split(",")[2], " ".join(cmd)
                df.loc[index, 'cov'] = float(result.stdout.split(",")[0])
            else:
                df.loc[index, 'cov'] = -1
        df.to_csv(file)


    processes = [Process(target=compute_cov_process, args=(file,)) for file in files]
    for process in processes:
        process.start()

