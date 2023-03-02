from tools.paper.method_packs import *
from tools.paper.gen_exp_dlmc import *

import subprocess
import tempfile
import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__)) + "/"

DATASET_DIR = "/datasets/"
TMP_FOLDER = "/tmp/"
SOURCE_ROOT = SCRIPT_DIR + "../../../"
TOOLS_DIR = SOURCE_ROOT + "tools/"
FILELISTS_PATH = TOOLS_DIR + "filelists"
EXPERIMENTS_FOLDER = TMP_FOLDER + "sp_reg_tiling_dlmc_test"

experiment_files = []

CPP_BENCH_BINARY = SOURCE_ROOT + "release-build/cpp_testbed/demo/SPMM_demo"

def run_experiment(experiment_file, matrix_file, output_file):
    print(experiment_file)
    subprocess.run([CPP_BENCH_BINARY, "-e", experiment_file, "-m", matrix_file, "-d", DATASET_DIR])


# Safe methods will not crash so we can run them all at once (saveing on matix loading)
safe_methods = all_intel_reported_methods["safe"]

# Unafe methods may crash so we run them one at a time so that if they crash it 
# doesn't impact other methods (this is primarily for ASpT which can be unstable)

BCOLS_TO_TEST = [32, 128, 256, 512]
NUM_THREADS_TO_TEST = [1, 8]

experiment_files.append(gen_dlmc_bench_exp("AVX512", safe_methods, "dlmc_60_to_95.txt", 
                                           BCOLS_TO_TEST, NUM_THREADS_TO_TEST, 
                                           output_path=EXPERIMENTS_FOLDER, 
                                           filelist_path=FILELISTS_PATH,
                                           return_full_path=True))

print(experiment_files)

for experiment in experiment_files:
    run_experiment(experiment, 
    "dlmc/transformer/random_pruning/0.8/body_decoder_layer_3_encdec_attention_multihead_attention_q_fully_connected.smtx",
    "result.txt")
