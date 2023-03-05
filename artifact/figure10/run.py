from tools.paper.method_packs import *
from tools.experiment_runner import *
from tools import experiment_runner
from sbench.loaders.filelist import FilelistPathIterator

from artifact.utils import *

import subprocess
import tempfile
import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__)) + "/"

MATRIX_FILELIST = SCRIPT_DIR + "filelist.txt"

def run_dlmc_experiments(method_packs, outfile, scalar_type, threads_to_test, datatransform):
    BCOLS_TO_TEST = [32, 128, 256, 512]
    experiment_files = []
    experiment_runner.append = False

    for method_pack in method_packs:
        experiment_files.append(
            gen_dlmc_exp_file(method_pack, BCOLS_TO_TEST, threads_to_test, MATRIX_FILELIST,
                              datatransform=datatransform))

    for matrix_file in FilelistPathIterator(MATRIX_FILELIST):
        run_sp_reg(BCOLS_TO_TEST, threads_to_test, matrix_file, outfile, scalar_type)

        for experiment in experiment_files:
            run_experiment(experiment, matrix_file, outfile, scalar_type)

# only run mkl_dense once, we will merge the files after
run_dlmc_experiments([mkl_dense], RESULTS_DIR + "figure10_transformed.csv", "float", [1], datatransform=True)
run_dlmc_experiments([], RESULTS_DIR + "figure10_not_transformed.csv", "float", [1], datatransform=False)