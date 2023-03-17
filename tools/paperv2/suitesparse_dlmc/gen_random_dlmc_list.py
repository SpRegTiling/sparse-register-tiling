from tools.paperv2.dlmc.utils import read_cache
import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__)) + "/"
import random

random.seed(42)

df = read_cache("cascade", "all", bcols=128, threads=20)
print(df)
num_matrices = 500

matrix_list = df["MatrixPath"].tolist()
random.shuffle(matrix_list)
print(matrix_list[:num_matrices])

with open(f"{SCRIPT_DIR}/dlmc_list.txt", "w+") as f:
    for m in matrix_list[:num_matrices]:
        f.write(m)
        f.write('\n')
