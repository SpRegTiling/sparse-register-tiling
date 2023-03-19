import glob
import sys
import numpy as np

from tools.paper.plotting.plot_utils import plot_save
from tools.paperv2.dlmc.utils import *

from matplotlib import rc, rcParams
from tools.paper.plotting.plot_utils import *
from scipy import stats

from tools.experiment_runner import *
from tools import experiment_runner
from tools.paper.configs import neon_nano_from_name

experiment_runner.DATASET_DIR = DATASET_DIR
os.makedirs("/tmp/experiment_scripts", exist_ok=True)

adfs = []
for bcol in [32, 128, 256, 512]:
    adfs.append(read_cache("raspberrypi", "all", bcols=bcol, threads=4))
adf = pd.concat(adfs)


for bcols in [32, 128, 256, 512]:
    for threads in [1, 4]:
        bcols = 32
        threads = 1
        outfile = RESULTS_DIR + f"/pi_rerun_{bcols}_{threads}.csv"
        bash_script = f"/tmp/experiment_scripts/pi_rerun_{bcols}_{threads}.sh"
        set_bash_script(bash_script)

        df = filter(adf, Bcols=bcols, numThreads=threads)
        for index, row in adf.iterrows():
            run_sp_reg_single(row['Bcols'], row['numThreads'], row["MatrixPath"], outfile,
                            "float", [neon_nano_from_name(row['orig_name|Sp. Reg.'], row['config|Sp. Reg.'])])
