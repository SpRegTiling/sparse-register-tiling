from tools.paper.method_packs import *
from tools.experiment_runner import *
import re
import subprocess
import tempfile
import csv
import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__)) + "/"
import argparse
import pathlib

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-m,--matrix', dest='matrix',
                    help='matrix file',
                    type=pathlib.Path,
                    required=True)

parser.add_argument('-t,--threads', dest='threads',
                    help='number of threads to use',
                    type=int,
                    required=True)

parser.add_argument('-b,--bcols', dest='bcols',
                    help='number of bcolumns to use',
                    type=int,
                    required=True)

parser.add_argument('-o,--outcsv', dest='outcsv',
                    help='csv to right the results to',
                    type=str,
                    required=True)

parser.add_argument('-d,--dense', dest='dense',
                    help='benchmark against mkl dense, if the matrix is very large this may cause your machine to hang',
                    action="store_true",
                    default=False)

args = parser.parse_args()

os.makedirs("/tmp/experiment_scripts", exist_ok=True)

run_sp_reg([args.bcols], [args.threads], str(args.matrix.resolve()), args.outcsv, 'float', extra_args=['-b', str(args.bcols)])

baselines = mkl_sparse
if args.dense:
    baselines += mkl_dense

baslines_exp = gen_dlmc_exp_file(baselines, [args.bcols], [args.threads], "no-filelist.txt")
run_experiment(baslines_exp, str(args.matrix.resolve()), args.outcsv, 'float', extra_args=['-b', str(args.bcols)])

print("======== Results ========")
with open(args.outcsv, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if re.search(r'M[0-9]N[0-9]', row['name']): row['name'] = "Sp. Reg."
        print(f'{row["name"]:10}: {round(float(row["time median"]), 2)} us  {row["correct"]}')