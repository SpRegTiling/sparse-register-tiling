from tools.paper.method_packs import *
from tools.experiment_runner import *
from sbench.loaders.filelist import FilelistPathIterator

from artifact.utils import *

#
#   NOTE: this file currenlty does not actually run the experiment but instead creates a 
#      bash script that be used to run the experiment, see `set_bash_script` for name
#

import subprocess
import tempfile
import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__)) + "/"
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--mtx_range', dest='mtx_range',
                    help='run a subset of the dataset')
parser.add_argument('--threads', dest='threads',
                    help='thread counts to run')

args = parser.parse_args()


os.makedirs("/tmp/experiment_scripts", exist_ok=True)
bash_script = f"/tmp/experiment_scripts/suitesparse_{str(args.mtx_range)}_{str(args.threads)}.sh"
set_bash_script(bash_script)
set_timeout("8m")

mtx_range = None
if args.mtx_range is not None:
    mtx_range = [int(x) for x in args.mtx_range.split("_")]
    assert len(mtx_range) == 2

threads = [20]
if args.threads is not None:
    threads = [int(x) for x in args.threads.split("_")]

def run_dlmc_experiments(method_packs, outfile, scalar_type, threads_to_test):
    experiment_files = []
    MATRIX_FILELIST = "suitesparse.txt"

    BCOLS_TO_TEST = [128]

    for method_pack in method_packs:
        experiment_files.append(
            gen_dlmc_exp_file(method_pack, BCOLS_TO_TEST, threads_to_test, MATRIX_FILELIST))

    for matrix_file in FilelistPathIterator(FILELISTS_PATH + MATRIX_FILELIST, range=mtx_range):
        run_sp_reg(BCOLS_TO_TEST, threads_to_test, matrix_file, outfile, scalar_type)
        for experiment in experiment_files:
            run_experiment(experiment, matrix_file, outfile, scalar_type)

mtx_range_str = "all" if mtx_range is None else "_".join([str(x) for x in mtx_range])
threads_str = "_".join([str(x) for x in threads])
outfile = f"suitesparse_double_results2_{mtx_range_str}__{threads_str}.csv"

run_dlmc_experiments([mkl_sparse], RESULTS_DIR + outfile, "double", threads)

print(bash_script)