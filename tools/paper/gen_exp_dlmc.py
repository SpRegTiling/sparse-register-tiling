from tools.paper.configs import nano, csb, nano_from_name

from collections.abc import Iterable
from collections import defaultdict
import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
import yaml
import shutil

from tools.paper.method_packs import method_packs
GENERATED_DIR = SCRIPT_DIR + "/experiments/generated/dlmc/"

def flatten(coll):
    for i in coll:
        if isinstance(i, Iterable) and not isinstance(i, str):
            for subc in flatten(i):
                yield subc
        else:
            yield i


def gen_dlmc_bench_exp(arch, test_methods, filelist, b_cols, num_threads, 
                       output_path=GENERATED_DIR, suffix = "", 
                       filelist_path="../../../../../../filelists",
                       return_full_path=False):
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
        "filelist": f'{filelist_path}/{filelist_name}.txt',
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

    print("arch", baseline_methods)

    experiment_file = f'{filelist_name}_{arch}{suffix}.yaml'
    dir = f'{output_path}/{arch}/yamls/'
    os.makedirs(dir, exist_ok=True)

    with open(dir + experiment_file, 'w+') as f:
        yaml.dump({
            "options": options,
            "tuning": {
                "parameter_grids": tuning_grids,
            },
            "matrices": matrices,
            "methods": baseline_methods + test_methods
        }, f)

    if return_full_path:
        return dir + experiment_file
    else:
        return experiment_file


def gen_run_script(files, script_name, is_aspt, part):
    files = flatten(files)
    os.makedirs(os.path.dirname(script_name), exist_ok=True)
    with open(script_name, 'w+') as f:
        f.write('SCRIPT_PATH=$(realpath $0)\n')
        f.write('SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")\n')
        if not is_aspt:
            for file in files:
                f.write(f"/bin/bash $1 $SCRIPT_DIR/yamls/{file} $2 {part}\n")
        else:
            for file in files:
                f.write(f"/bin/bash $1 $SCRIPT_DIR/yamls/{file} $2 {part}\n")


if __name__ == "__main__":
    sub_dir = 'dlmc'
    pack_names = set()
    files = defaultdict(lambda: defaultdict(lambda: []))
    dlmc_parts = 5

    shutil.rmtree(GENERATED_DIR, ignore_errors=True)
    os.makedirs(GENERATED_DIR, exist_ok=True)

    # All-DLMC Experiments
    for arch in ["AVX512"]:
        max_threads_by_arch = {
            "AVX512": [20],
            "NEON": [4]
        }

        for max_thread_count in max_threads_by_arch[arch]:
            n_threads = {
                4: [1, 4],
                8: [1, 8],
                20: [20],
                32: [1, 16, 32],
                64: [1, 16, 32, 64],
            }

            all_files = []
            for pack_name, methods in method_packs[arch].items():
                pack_names.add(pack_name)

                for part in range(1, dlmc_parts+1):
                    files[(arch, max_thread_count)][(pack_name, part)] = [
                        gen_dlmc_bench_exp(arch, methods, f'dlmc_part{part}',
                                        b_cols=[32, 256],
                                        num_threads=n_threads[max_thread_count],
                                        suffix=f"{pack_name}_small_bcols_{max_thread_count}"),
                        gen_dlmc_bench_exp(arch, methods, f'dlmc_part{part}',
                                        b_cols=[128, 512],
                                        num_threads=n_threads[max_thread_count],
                                        suffix=f"{pack_name}_large_bcols_{max_thread_count}"),
                    ]

    for (arch, n_threads), packs in files.items():
        files_for_parts = [[] for x in range(dlmc_parts)]
        aspt_files_for_part = [[] for x in range(dlmc_parts)]

        # Create scripts for each pack
        for (pack_name, part), files in packs.items():
            gen_run_script(files, f"{GENERATED_DIR}/{arch}/{pack_name}_part{part}.sh", True, part)
            if "aspt" in pack_name:
                aspt_files_for_part[part-1] += files
            else:
                files_for_parts[part-1] += files

        for i in range(1, dlmc_parts+1):
            gen_run_script(files_for_parts[i-1], f"{GENERATED_DIR}/{arch}/all_part{i}.sh", True, i)

        for i in range(1, dlmc_parts+1):
            gen_run_script(aspt_files_for_part[i-1], f"{GENERATED_DIR}/{arch}/all_aspt_part{i}.sh", True, i)




# for pack_name in pack_names:
#     gen_cluster_scripts(sub_dir, run_all_script_name + "_" + pack_name)
# gen_cluster_scripts(sub_dir, run_all_script_name)
