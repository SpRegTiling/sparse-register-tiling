from collections.abc import Iterable
from collections import defaultdict
import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
import yaml
import shutil

from tools.paper.method_packs import method_packs

sub_dir = 'dlmc'
pack_names = set()
files = defaultdict(lambda: defaultdict(lambda: []))
dlmc_parts = 5

GENERATED_DIR = SCRIPT_DIR + "/experiments/generated/random/"
shutil.rmtree(GENERATED_DIR, ignore_errors=True)
os.makedirs(GENERATED_DIR, exist_ok=True)


def gen_random_sweep_exp(arch, test_methods, filelist, b_cols, num_threads, suffix = ""):
    filelist_name = 'random_sweep_30_100'

    if suffix != "" and suffix[0] != "_":
        suffix = "_" + suffix

    options = {
        "profile": True,
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

    experiment_file = f'{filelist_name}_{arch}{suffix}.yaml'
    dir = f'{GENERATED_DIR}/{arch}/yamls/'
    os.makedirs(dir, exist_ok=True)

    with open(dir + experiment_file, 'w+') as f:
        yaml.dump({
            "options": options,
            "matrices": matrices,
            "methods": test_methods
        }, f)

    return experiment_file



from collections import Iterable
def flatten(coll):
    for i in coll:
        if isinstance(i, Iterable) and not isinstance(i, str):
            for subc in flatten(i):
                yield subc
        else:
            yield i


# All-DLMC Experiments
for arch in ["AVX512", "NEON"]:
    max_threads_by_arch = {
        "AVX2": [32, 64],
        "AVX512": [20],
        "NEON": [4]
    }

    for max_thread_count in max_threads_by_arch[arch]:
        n_threads = {
            4: [1, 4],
            8: [1, 8],
            32: [1, 16, 32],
            20: [1, 16, 20],
            64: [1, 16, 32, 64],
        }

        all_files = []
        for pack_name, methods in method_packs[arch].items():
            pack_names.add(pack_name)

            files[(arch, max_thread_count)][(pack_name)] = [
                gen_random_sweep_exp(arch, methods, f'random_sweep',
                                   b_cols=[32, 128, 256],
                                   num_threads=n_threads[max_thread_count],
                                   suffix=f"{pack_name}_small_bcols_{max_thread_count}"),
                gen_random_sweep_exp(arch, methods, f'random_sweep',
                                   b_cols=[512, 1024],
                                   num_threads=n_threads[max_thread_count],
                                   suffix=f"{pack_name}_large_bcols_{max_thread_count}"),
            ]


def gen_run_script(files, script_name):
    files = flatten(files)
    os.makedirs(os.path.dirname(script_name), exist_ok=True)
    with open(script_name, 'w+') as f:
        f.write('SCRIPT_PATH=$(realpath $0)\n')
        f.write('SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")\n')
        for file in files:
            f.write(f"/bin/bash $1 $SCRIPT_DIR/yamls/{file} $2\n")


for (arch, n_threads), packs in files.items():
    all_files = []

    # Create scripts for each pack
    for (pack_name), files in packs.items():
        gen_run_script(files, f"{GENERATED_DIR}/{arch}/{pack_name}_randomsweep.sh")
        all_files.append(files)

    gen_run_script(flatten(all_files), f"{GENERATED_DIR}/{arch}/all_randomsweep.sh")

