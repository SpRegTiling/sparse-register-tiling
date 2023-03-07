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
set_bash_script("/tmp/experiment_scripts/figure10.sh")

MATRIX_FILELIST = SCRIPT_DIR + "filelist.txt"

def run_dlmc_experiments(method_packs, outfile, scalar_type, threads_to_test, datatransform, methods_to_test):
    BCOLS_TO_TEST = [32, 128, 256, 512]
    experiment_files = []
    experiment_runner.append = False

    for method_pack in method_packs:
        experiment_files.append(
            gen_dlmc_exp_file(method_pack, BCOLS_TO_TEST, threads_to_test, MATRIX_FILELIST,
                              datatransform=datatransform))

    for i, matrix_file in enumerate(FilelistPathIterator(MATRIX_FILELIST)):
        run_sp_reg(BCOLS_TO_TEST, threads_to_test, matrix_file, outfile, scalar_type, 
                   methods_to_test=methods_to_test, method_idx=i, datatransform=datatransform)

        for experiment in experiment_files:
            run_experiment(experiment, matrix_file, outfile, scalar_type)


transformed_methods = defaultdict(lambda: {})
not_transformed_methods = defaultdict(lambda: {})
with open(SCRIPT_DIR + 'ordered_methods.txt', 'r') as f:
    ordered_methods = [x.rstrip() for x in f.readlines()]
    bcols = [32, 128, 256, 512]
    for i, bcol in enumerate(bcols):
        transformed_methods[1][bcol] = ordered_methods[(2*i + 0) * 18: (2*i + 1) * 18]
        not_transformed_methods[1][bcol] = ordered_methods[(2*i + 1) * 18: (2*i + 2) * 18]


# only run mkl_dense once, we will merge the files after
run_dlmc_experiments([mkl_dense], RESULTS_DIR + "figure10_transformed.csv", "float", [1], 
                     datatransform=True, methods_to_test=transformed_methods)
run_dlmc_experiments([], RESULTS_DIR + "figure10_not_transformed.csv", "float", [1], 
                     datatransform=False, methods_to_test=not_transformed_methods)