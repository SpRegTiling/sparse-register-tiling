from tools.paper.method_packs import *
from tools.experiment_runner import *
from tools import experiment_runner
from sbench.loaders.filelist import FilelistPathIterator

from artifact.utils import *

import subprocess
import tempfile
import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__)) + "/"

#
#   NOTE: this file currenlty does not actually run the experiment but instead creates a 
#      bash script that be used to run the experiment, see `set_bash_script` for name
#

os.makedirs("/tmp/experiment_scripts", exist_ok=True)
set_bash_script("/tmp/experiment_scripts/figure11.sh")


MATRIX_FILELIST = SCRIPT_DIR + "filelist.txt"

def run_dlmc_experiments(method_packs, outfile, scalar_type, threads_to_test):
    BCOLS_TO_TEST = [32, 128, 256, 512]
    experiment_files = []
    experiment_runner.append = False

    for method_pack in method_packs:
        experiment_files.append(
            gen_dlmc_exp_file(method_pack, BCOLS_TO_TEST, threads_to_test, MATRIX_FILELIST,
                              profile=True))

    for matrix_file in FilelistPathIterator(MATRIX_FILELIST, datatset_dir=DATASET_DIR):
        run_sp_reg(BCOLS_TO_TEST, threads_to_test, matrix_file, outfile, scalar_type)
        for experiment in experiment_files:
            run_experiment(experiment, matrix_file, outfile, scalar_type)

# only run mkl_dense once, we will merge the files after
# nano_from_name('M8N3_NKM_LB_TLB128_SA_orig') is the sparse register tiling method (not using heursitic)
method_packs = [mkl_sparse]
run_dlmc_experiments(method_packs, RESULTS_DIR + "figure11.csv", "float", [1])
