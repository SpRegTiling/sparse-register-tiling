import sys

from tools.paper.configs import nano, csb, nano_from_name

from collections.abc import Iterable
from collections import defaultdict
import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
import yaml
import shutil

from tools.paper.method_packs import method_packs

pack_names = set()
files = defaultdict(lambda: defaultdict(lambda: []))
dlmc_parts = 5

filelist = sys.argv[1]
print(filelist)

GENERATED_DIR = SCRIPT_DIR + f"/experiments/generated/{filelist.replace('.txt', '')}/"
shutil.rmtree(GENERATED_DIR, ignore_errors=True)
os.makedirs(GENERATED_DIR, exist_ok=True)


def flatten(coll):
    for i in coll:
        if isinstance(i, Iterable) and not isinstance(i, str):
            for subc in flatten(i):
                yield subc
        else:
            yield i


def gen_bench_exp(arch, test_methods, filelist, b_cols, num_threads, suffix = ""):
    filelist_name = filelist.split("/")[-1].replace(".txt", "")

    if suffix != "" and suffix[0] != "_":
        suffix = "_" + suffix

    options = {
        "profile": False,
        "scalar_type": "float",
        "b_cols": b_cols,
        "n_threads": num_threads,
        "output_file": f"results/{filelist_name}_{arch}{suffix}.csv",
        "save_tuning_results": False,
        "expand_config_parameters": [ "m_tile", "k_tile", "n_tile", "tiling_strategy", "sparse_a", "beta_10x" ]
    }

    matrices = {
        "filelist": f'../../../../../../filelists/{filelist_name}.txt',
    }

    tuning_grids = [
        {
            "name": "SOP4",
            "m_tile": [ 32, 64 ],
            "k_tile": [ 512, 256, 128, 64 ],
            "n_tile": [ 64, 128, 256 ],
            "tiling_strategy": [ 0 ],
        },
        {
            "name": "SOP8",
            "m_tile": [ 8, 32, 64 ],
            "k_tile": [ 512, 256, 128, 64 ],
            "n_tile": [ 32, 64, 128 ],
            "tiling_strategy": [ 0 ],
        }
    ]

    if "AVX" in arch:
        baseline_methods = [

        ]
    elif "NEON" in arch:
        baseline_methods = [

        ]
    else:
        assert False, "Unknown architecture"

    experiment_file = f'{filelist_name}_{arch}{suffix}.yaml'
    dir = f'{GENERATED_DIR}/{arch}/yamls/'
    os.makedirs(dir, exist_ok=True)

    with open(dir + experiment_file, 'w+') as f:
        yaml.safe_dump({
            "options": options,
            "tuning": {
                "parameter_grids": tuning_grids,
            },
            "matrices": matrices,
            "methods": baseline_methods + test_methods
        }, f)

    return experiment_file


# All-DLMC Experiments
for arch in ["AVX512"]:
    max_threads_by_arch = {
        "AVX2": [32, 64],
        "AVX512": [20],
        "NEON": [4]
    }

    for max_thread_count in max_threads_by_arch[arch]:
        n_threads = {
            4: [1, 4],
            8: [1, 8],
            20: [20],
            64: [1, 16, 32, 64],
        }

        all_files = []
        for pack_name, methods in method_packs[arch].items():
            pack_names.add(pack_name)

            files[(arch, max_thread_count)][pack_name] = [
                gen_bench_exp(arch, methods, filelist,
                              b_cols=[32, 128],
                              num_threads=n_threads[max_thread_count],
                              suffix=f"{pack_name}_small_bcols_{max_thread_count}"),
                # gen_bench_exp(arch, methods, filelist,
                #               b_cols=[512, 1024],
                #               num_threads=n_threads[max_thread_count],
                #               suffix=f"{pack_name}_large_bcols_{max_thread_count}"),
            ]


def gen_run_script(files, script_name):
    files = flatten(files)
    os.makedirs(os.path.dirname(script_name), exist_ok=True)
    with open(script_name, 'w+') as f:
        f.write('SCRIPT_PATH=$(realpath $0)\n')
        f.write('SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")\n')
        for file in files:
            f.write(f"/bin/bash $1 $SCRIPT_DIR/yamls/{file} $2 $3\n")


for (arch, n_threads), packs in files.items():
    files_for_parts = [[] for x in range(dlmc_parts)]

    # Create scripts for each pack
    all_files = []
    for pack_name, files in packs.items():
        gen_run_script(files, f"{GENERATED_DIR}/{arch}/{pack_name}.sh")
        all_files += files

    gen_run_script(all_files, f"{GENERATED_DIR}/{arch}/all.sh")



# for pack_name in pack_names:
#     gen_cluster_scripts(sub_dir, run_all_script_name + "_" + pack_name)
# gen_cluster_scripts(sub_dir, run_all_script_name)
