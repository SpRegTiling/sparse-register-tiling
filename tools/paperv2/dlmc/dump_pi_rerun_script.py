import glob
import sys
import numpy as np

from tools.paper.plotting.plot_utils import plot_save
from tools.paperv2.dlmc.utils import *

from matplotlib import rc, rcParams
from tools.paper.plotting.plot_utils import *
from scipy import stats

from tools import experiment_runner
from tools.paper.configs import neon_nano_from_name

experiment_runner.DATASET_DIR = "/home/pi/"
experiment_runner.EXPERIMENTS_FOLDER = "/tmp/pi_rerun_scripts/yamls/"
experiment_runner.CPP_BENCH_BINARY = "/home/pi/spmm-nano-bench/release-build/cpp_testbed/demo/SPMM_demo"
os.makedirs("/tmp/pi_rerun_scripts/", exist_ok=True)
os.makedirs("/tmp/pi_rerun_scripts/yamls/", exist_ok=True)

adfs = []
for bcol in [32, 128, 256, 512]:
    adfs.append(read_cache("raspberrypi", "all", bcols=bcol, threads=4))
adf = pd.concat(adfs)

baselines = [
    {
        "name": "XNN",
        "method_id": "xnn",
    },
    {
        "name": "ARMCL",
        "method_id": "armcl_dense",
    }
]

for bcols in [256]:
    for threads in [1, 4]:
        outfile = f"/home/pi/pi_rerun_{bcols}_{threads}.csv"
        bash_script = f"/tmp/pi_rerun_scripts/pi_rerun_{bcols}_{threads}.sh"
        experiment_runner.set_bash_script(bash_script)

        df = filter(adf, Bcols=bcols, numThreads=threads)
        for index, row in adf.iterrows():
            experiment_runner.run_sp_reg_single(row['Bcols'], row['numThreads'], row["MatrixPath"], outfile,
                            "float", [neon_nano_from_name(row['orig_name|Sp. Reg.'], row['config|Sp. Reg.'])])
