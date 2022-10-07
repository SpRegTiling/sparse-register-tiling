om spmm_benchmarks.loaders.dlmc import DLMCPathIterator

from tools.paper.configs import nano, csb
from tools.paper.clusters import gen_cluster_scripts

from collections.abc import Iterable
import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
import yaml

import numpy as np
import random
import math

sub_dir = 'dlmc'
run_all_script_name = 'all_dlmc'


def flatten(coll):
    for i in coll:
        if isinstance(i, Iterable) and not isinstance(i, str):
            for subc in flatten(i):
                yield subc
        else:
            yield i


dlmc_paths = list(DLMCPathIterator())
print(dlmc_paths)

random.seed(42)
random.shuffle(dlmc_paths)
parts = 5

part_size = math.ceil(len(dlmc_paths) / parts)
for i in range(parts):
    with open(SCRIPT_DIR + f"/../filelists/dlmc_part{i+1}.txt", 'w+') as f:
        for path in dlmc_paths[i*part_size:(i+1)*part_size]:
            f.write("dlmc" + path.split("dlmc")[-1] + "\n")
