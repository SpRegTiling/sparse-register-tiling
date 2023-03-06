from tools.paper.gen_exp_dlmc import *
from collections import defaultdict

import subprocess
import hashlib
import json
import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__)) + "/"

BASH_SCRIPT=None
def set_bash_script(bash_script):
    global BASH_SCRIPT
    try:
        os.remove(bash_script)
    except OSError:
        pass
    BASH_SCRIPT = bash_script
    with open(bash_script, "a+") as f:
            f.write("#!/bin/bash\n")


DATASET_DIR = "/datasets/"
TMP_FOLDER = "/tmp/"
SOURCE_ROOT = SCRIPT_DIR + "../"
TOOLS_DIR = SOURCE_ROOT + "tools/"
FILELISTS_PATH = TOOLS_DIR + "filelists/"
EXPERIMENTS_FOLDER = TMP_FOLDER + "sp_reg_tiling_dlmc_test"

CPP_BENCH_BINARY = SOURCE_ROOT + "release-build/cpp_testbed/demo/SPMM_demo"

def gen_dlmc_exp_file(methods, bcols, threads, matrix_filelist, datatransform=True, profile=False):
    id = hashlib.md5((f'{str(methods)}:{bcols}:{threads}:{profile}:{datatransform}')\
                     .encode('utf-8')).hexdigest()[-5:]
    return gen_dlmc_bench_exp("AVX512", methods, matrix_filelist, 
                        bcols, threads, 
                        output_path=EXPERIMENTS_FOLDER, 
                        filelist_path=FILELISTS_PATH,
                        return_full_path=True,
                        datatransform=datatransform,
                        extra_csv_columns=["required_storage"],
                        profile=profile,
                        suffix=id)


append = False
def run_experiment(experiment_file, matrix_file, output_file, scalar_type, append_override=None, extra_args=[]):
    global append
    command = [CPP_BENCH_BINARY, "-e", experiment_file, "-m", matrix_file, "-d", DATASET_DIR, "-s", scalar_type]
    command += ['-o', output_file]
    command += extra_args
    append = append_override if append_override is not None else append
    if append: command += ['-a']
    print(" ".join(command))
    if BASH_SCRIPT is None:
        subprocess.run(command)
    else:
        with open(BASH_SCRIPT, "a+") as f:
            f.write(" ".join(command) + "\n")
    append = True


loaded_heuristics = {}

def parse_sparsity_range(range_str):
    return tuple(float(x) for x in range_str[1:-1].split(","))

def matrix_sparsity(matrix_file):
    with open(matrix_file, "r+") as f:
        firstline = f.readline()
        if "%%MatrixMarket" in firstline:
            print(firstline)
            while "%" in (line := f.readline()): print(line)
            print(line)
            rows, cols, nnz = [int(x) for x in line.rstrip().split(" ")]
        else:
            rows, cols, nnz = [int(x) for x in firstline.split(",")]
        return 1 - (nnz / (rows * cols))

def run_sp_reg(bcols, threads, matrix_file, output_file, scalar_type, methods_to_test=None, method_idx=None):
    global loaded_heuristics

    if methods_to_test is None:
        for thread_count in threads:
            if not thread_count in loaded_heuristics:
                loaded_heuristics[thread_count] = load_heuristic(thread_count)
            heuristic = loaded_heuristics[thread_count]

            sparsity = round(matrix_sparsity(matrix_file), 2)
            for sparsity_range, h in heuristic.items():
                if sparsity_range[0] < sparsity and sparsity <= sparsity_range[1]:
                    for bcol in bcols:
                        run_experiment(h[bcol], matrix_file, output_file, scalar_type, extra_args=['-z'])
    else:
        for thread_count in threads:
            for bcol in bcols:
                if method_idx is not None:
                    x = methods_to_test[thread_count][bcol][method_idx]
                else:
                    x = methods_to_test[thread_count][bcol]
                exp_file = gen_dlmc_exp_file([nano_from_name("AVX512", x)], 
                                [int(bcol)], [threads], "no_filelist.txt")
        

def load_heuristic(threads):
    if threads == 1:
        filename = "empirical_heuristic_single_threaded.json"
    else:
        filename = "empirical_heuristic_multi_threaded.json"

    with open(SCRIPT_DIR + filename, 'r') as f:
        heuristic_dict = json.load(f)
        return gen_experiment_files_for_heuristic(heuristic_dict, threads)

def gen_experiment_files_for_heuristic(heuristic_dict, threads):
    heuristic_experiment_files_dict = defaultdict(lambda: {})
    for sparsity_range, h in heuristic_dict.items():
        sparsity_range = parse_sparsity_range(sparsity_range)
        for bcol, methods in h.items():
            exp_file = gen_dlmc_exp_file([nano_from_name("AVX512", x) for x in methods], 
                                         [int(bcol)], [threads], "no_filelist.txt")
            heuristic_experiment_files_dict[sparsity_range][int(bcol)] = exp_file
    return heuristic_experiment_files_dict