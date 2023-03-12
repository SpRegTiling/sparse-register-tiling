import glob
import os
import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__)) + "/"

with open(SCRIPT_DIR + "/filelists/suitesparse.txt", "w+") as f:
    for subdir in list(os.walk("/datasets/suitesparse"))[1:]:
        print(subdir)
        matrix_name = subdir[0].split("/")[-1]
        if "tar" in matrix_name: continue
        f.write(f"suitesparse/{matrix_name}/{matrix_name}.mtx\n")
