import os
import glob
import argparse
import subprocess

OBJDUMP_CMD = 'objdump --no-show-raw-insn -C -S -d '
SPMM_FOLDER_PATH = '/cpp_testbed/demo/CMakeFiles/SPMM_demo.dir/__/__/xformers/sparse/backends/cpu/spmm/'

parser = argparse.ArgumentParser(description='Dump disassembly for a specific kernel')
parser.add_argument('build_folder', help='an integer for the accumulator')
parser.add_argument('-k', dest='kernel', help='specific kernel to dump')
parser.add_argument('-f', dest='file', help='specific file to dump')

args = parser.parse_args()

if args.file is None:
    files = glob.glob(args.build_folder + SPMM_FOLDER_PATH + "*.o")
else:
    files = [args.build_folder + SPMM_FOLDER_PATH + args.file + ".o"]

for file in files:
    out = subprocess.check_output(OBJDUMP_CMD + file, shell=True)
    print(out.decode('unicode_escape'))

print(files)
