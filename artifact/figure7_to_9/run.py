from tools.paper.method_packs import *
from tools.experiment_runner import *
from sbench.loaders.filelist import FilelistPathIterator

from artifact.utils import *


import subprocess
import tempfile
import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__)) + "/"

def run_dlmc_experiments(method_packs, outfile, scalar_type, threads_to_test):
    experiment_files = []
    MATRIX_FILELIST = "dlmc_60_to_95.txt"

    BCOLS_TO_TEST = [32, 128] #, 256, 512]

    for method_pack in method_packs:
        experiment_files.append(
            gen_dlmc_exp_file(method_pack, BCOLS_TO_TEST, threads_to_test, MATRIX_FILELIST))


    print(experiment_files)

    for matrix_file in FilelistPathIterator(FILELISTS_PATH + MATRIX_FILELIST, percentage=0.02):
        run_sp_reg(BCOLS_TO_TEST, threads_to_test, matrix_file, outfile, scalar_type)
        for experiment in experiment_files:
            run_experiment(experiment, matrix_file, outfile, scalar_type)

# Safe methods will not crash so we can run them all at once (saveing on matix loading)
safe_methods = all_intel_float_reference_methods["safe"]

# Unafe methods may crash so we run them one at a time so that if they crash it 
# doesn't impact other methods (this is primarily for ASpT which can be unstable)
unsafe_methods = all_intel_float_reference_methods["buggy"]

method_packs = [safe_methods] + [[x] for x in unsafe_methods]
run_dlmc_experiments(method_packs, RESULTS_DIR + "figure7_to_9_results.csv", "float", [1,8])