from tools.paper.configs import nano, csb
from tools.paper.clusters import gen_cluster_scripts

from collections.abc import Iterable
from collections import defaultdict
import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
import yaml
import shutil

sub_dir = 'mobilenet'
pack_names = set()
files = defaultdict(lambda: defaultdict(lambda: []))
dlmc_parts = 5

GENERATED_DIR = SCRIPT_DIR + "/experiments/generated/mobilenet/"
shutil.rmtree(GENERATED_DIR, ignore_errors=True)
os.makedirs(GENERATED_DIR, exist_ok=True)


def flatten(coll):
    for i in coll:
        if isinstance(i, Iterable) and not isinstance(i, str):
            for subc in flatten(i):
                yield subc
        else:
            yield i


def gen_dlmc_bench_exp(arch, test_methods, filelist, b_cols, num_threads, suffix = ""):
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

    print("arch", baseline_methods)

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
for arch in ["AVX2", "AVX512", "NEON"]:
    max_threads_by_arch = {
        "AVX2": [1],
        "AVX512": [32],
        "NEON": [4]
    }

    nrs_by_arch = {
        "AVX512": {
            4: [3,2],
            8: [2,1],
        },
        "AVX2": {
            4: [3,2],
            8: [2,1],
        },
        "NEON": {
            4: [3,2],
            8: [2,1],
        }
    }

    for max_thread_count in max_threads_by_arch[arch]:
        n_threads = {
            1: [1],
            4: [1, 4],
            8: [1, 8],
            32: [1, 16, 32],
            64: [1, 16, 32, 64],
        }

        method_packs = {
        "mkl": [
            {
                "name": "MKL_Dense",
                "method_id": "mkl_dense"
            },
            {
                "name": "MKL_Sparse",
                "method_id": "mkl",
                "options": {
                    "inspector": False
                }
            },
            {
                "name": "MKL_Sparse_IE",
                "method_id": "mkl",
                "options": {
                    "inspector": True
                }
            }
        ],
        "nano4_identity_NKM": [
            nano(arch, 4, nrs_by_arch[arch][4][1], "identity", "NKM"),
            nano(arch, 4, nrs_by_arch[arch][4][1], "identity", "NKM", load_balance=True, tlb_comp=64),
            nano(arch, 4, nrs_by_arch[arch][4][1], "identity", "NKM", load_balance=True, sparse_a=True),
            nano(arch, 4, nrs_by_arch[arch][4][1], "identity", "NKM", load_balance=True, sparse_a=True, packed=True),
            nano(arch, 4, nrs_by_arch[arch][4][1], "identity", "NKM", load_balance=True, sparse_a=True, tlb_comp=64),
            nano(arch, 4, nrs_by_arch[arch][4][1], "identity", "NKM", load_balance=True, sparse_a=True, tlb_comp=128),
            nano(arch, 4, nrs_by_arch[arch][4][0], "identity", "NKM"),
            nano(arch, 4, nrs_by_arch[arch][4][0], "identity", "NKM", load_balance=True, sparse_a=True, packed=True),
            nano(arch, 4, nrs_by_arch[arch][4][0], "identity", "NKM", load_balance=True, sparse_a=True, tlb_comp=64),
            nano(arch, 4, nrs_by_arch[arch][4][0], "identity", "NKM", load_balance=True, sparse_a=True, tlb_comp=128),
        ],
        "nano4_orig_NKM": [
            nano(arch, 4, nrs_by_arch[arch][4][1], "orig", "NKM",     load_balance=True),
            nano(arch, 4, nrs_by_arch[arch][4][1], "orig", "NKM",     load_balance=True, sparse_a=True, packed=True),
            nano(arch, 4, nrs_by_arch[arch][4][1], "orig", "NKM",     load_balance=True, sparse_a=True, tlb_comp=64),
            nano(arch, 4, nrs_by_arch[arch][4][0], "orig", "NKM",     load_balance=True),
            nano(arch, 4, nrs_by_arch[arch][4][0], "orig", "NKM",     load_balance=True, sparse_a=True, packed=True),
            nano(arch, 4, nrs_by_arch[arch][4][0], "orig", "NKM",     load_balance=True, sparse_a=True, tlb_comp=64),
        ],
        "nano8_orig_NKM":  [
            nano(arch, 8, nrs_by_arch[arch][8][1], "orig", "NKM"),
            nano(arch, 8, nrs_by_arch[arch][8][1], "orig", "NKM", load_balance=True),
            nano(arch, 8, nrs_by_arch[arch][8][1], "orig", "NKM", load_balance=True, packed=True),
            nano(arch, 8, nrs_by_arch[arch][8][1], "orig", "NKM", load_balance=True, tlb_comp=64),
            nano(arch, 8, nrs_by_arch[arch][8][1], "orig", "NKM", load_balance=True, sparse_a=True, tlb_comp=64),
            nano(arch, 8, nrs_by_arch[arch][8][1], "orig", "NKM", load_balance=True, sparse_a=True, tlb_comp=128),
            nano(arch, 8, nrs_by_arch[arch][8][0], "orig", "NKM"),
            nano(arch, 8, nrs_by_arch[arch][8][0], "orig", "NKM", load_balance=True),
            nano(arch, 8, nrs_by_arch[arch][8][0], "orig", "NKM", load_balance=True, packed=True),
            nano(arch, 8, nrs_by_arch[arch][8][0], "orig", "NKM", load_balance=True, sparse_a=True, tlb_comp=64),
            nano(arch, 8, nrs_by_arch[arch][8][0], "orig", "NKM", load_balance=True, sparse_a=True, tlb_comp=128),
        ],
        "nano8_alt_NKM":  [
            nano(arch, 8, nrs_by_arch[arch][8][1], "alt", "NKM"),
            nano(arch, 8, nrs_by_arch[arch][8][1], "alt", "NKM",  load_balance=True, sparse_a=True, tlb_comp=64),
            nano(arch, 8, nrs_by_arch[arch][8][1], "alt", "NKM",  load_balance=True, sparse_a=True, tlb_comp=64, packed=True),
            nano(arch, 8, nrs_by_arch[arch][8][0], "alt", "NKM"),
            nano(arch, 8, nrs_by_arch[arch][8][0], "alt", "NKM",  load_balance=True, sparse_a=True, tlb_comp=64),
            nano(arch, 8, nrs_by_arch[arch][8][0], "alt", "NKM",  load_balance=True, sparse_a=True, tlb_comp=64, packed=True),
        ],
        "nano4_identity_KNM": [
            nano(arch, 4, nrs_by_arch[arch][4][0], "identity", "KNM"),
            nano(arch, 4, nrs_by_arch[arch][4][0], "identity", "KNM", load_balance=True, tlb_comp=64, packed=True),
            nano(arch, 4, nrs_by_arch[arch][4][0], "identity", "KNM", load_balance=True, tlb_comp=64),
            nano(arch, 4, nrs_by_arch[arch][4][0], "identity", "KNM", load_balance=True, sparse_a=True),
            nano(arch, 4, nrs_by_arch[arch][4][0], "identity", "KNM", load_balance=True, sparse_a=True, tlb_comp=64),
            nano(arch, 4, nrs_by_arch[arch][4][0], "identity", "KNM", load_balance=True, sparse_a=True, tlb_comp=128),
            nano(arch, 4, nrs_by_arch[arch][4][1], "identity", "KNM"),
            nano(arch, 4, nrs_by_arch[arch][4][1], "identity", "KNM", load_balance=True, packed=True),
            nano(arch, 4, nrs_by_arch[arch][4][1], "identity", "KNM", load_balance=True, sparse_a=True, tlb_comp=48),
            nano(arch, 4, nrs_by_arch[arch][4][1], "identity", "KNM", load_balance=True, sparse_a=True, tlb_comp=64),
            nano(arch, 4, nrs_by_arch[arch][4][1], "identity", "KNM", load_balance=True, sparse_a=True, tlb_comp=128),
        ],
        "nano4_orig_KNM": [
            nano(arch, 4, nrs_by_arch[arch][4][0], "orig", "KNM",     load_balance=True),
            nano(arch, 4, nrs_by_arch[arch][4][0], "orig", "KNM",     load_balance=True, sparse_a=True, tlb_comp=64),
            nano(arch, 4, nrs_by_arch[arch][4][1], "orig", "KNM",     load_balance=True),
            nano(arch, 4, nrs_by_arch[arch][4][1], "orig", "KNM",     load_balance=True, sparse_a=True, tlb_comp=64),
        ],
        "nano8_orig_KNM":  [
            nano(arch, 8, nrs_by_arch[arch][8][0], "orig", "KNM"),
            nano(arch, 8, nrs_by_arch[arch][8][0], "orig", "KNM", load_balance=True),
            nano(arch, 8, nrs_by_arch[arch][8][0], "orig", "KNM", load_balance=True, packed=True),
            nano(arch, 8, nrs_by_arch[arch][8][0], "orig", "KNM", load_balance=True, tlb_comp=64),
            nano(arch, 8, nrs_by_arch[arch][8][0], "orig", "KNM", load_balance=True, sparse_a=True, tlb_comp=64),
            nano(arch, 8, nrs_by_arch[arch][8][0], "orig", "KNM", load_balance=True, sparse_a=True, tlb_comp=128),
            nano(arch, 8, nrs_by_arch[arch][8][1], "orig", "KNM"),
            nano(arch, 8, nrs_by_arch[arch][8][1], "orig", "KNM", load_balance=True),
            nano(arch, 8, nrs_by_arch[arch][8][1], "orig", "KNM", load_balance=True, packed=True),
            nano(arch, 8, nrs_by_arch[arch][8][1], "orig", "KNM", load_balance=True, sparse_a=True, tlb_comp=64),
            nano(arch, 8, nrs_by_arch[arch][8][1], "orig", "KNM", load_balance=True, sparse_a=True, tlb_comp=128),
        ],
        "nano8_alt_KNM":  [
            nano(arch, 8, nrs_by_arch[arch][8][0], "alt", "KNM"),
            nano(arch, 8, nrs_by_arch[arch][8][0], "alt", "KNM",  load_balance=True, sparse_a=True, tlb_comp=64),
            nano(arch, 8, nrs_by_arch[arch][8][1], "alt", "KNM"),
            nano(arch, 8, nrs_by_arch[arch][8][1], "alt", "KNM",  load_balance=True, sparse_a=True, tlb_comp=64),
        ],
        "nano_tuned_NKM":  [
            nano(arch, 4, nrs_by_arch[arch][4][0], "identity", "NKM", load_balance=True, tune="SOP4"),
            nano(arch, 4, nrs_by_arch[arch][4][1], "orig",     "NKM", load_balance=True, tune="SOP4"),
            nano(arch, 8, nrs_by_arch[arch][8][0], "orig",     "NKM", load_balance=True, tune="SOP4"),
            nano(arch, 8, nrs_by_arch[arch][8][0], "alt",      "NKM", load_balance=True, tune="SOP4")
        ],
        # "nano_tuned_KNM":  [
        #     nano(arch, 4, 4, "identity", "KNM", load_balance=True, tune="SOP4"),
        #     nano(arch, 4, 4, "orig",     "KNM", load_balance=True, tune="SOP4"),
        #     nano(arch, 8, 2, "orig",     "KNM", load_balance=True, tune="SOP4"),
        #     nano(arch, 8, 2, "alt",      "KNM", load_balance=True, tune="SOP4")
        # ],

        # "csb": [
        #     csb(arch, "CSR", 32),
        #     csb(arch, "CSR", 32, sparse_a=True, tlb_comp=64),
        #     csb(arch, "CSR", 64),
        #     csb(arch, "CSR", 64, sparse_a=True, tlb_comp=64),
        # ]
        }

        all_files = []
        for pack_name, methods in method_packs.items():
            pack_names.add(pack_name)

            files[(arch, max_thread_count)][(pack_name, 1)] = [gen_dlmc_bench_exp(arch, methods, f'mobilenet_80',
                                        b_cols=-1,
                                        num_threads=n_threads[max_thread_count],
                                        suffix=f"{pack_name}_{max_thread_count}")]


def gen_run_script(files, script_name):
    files = flatten(files)
    os.makedirs(os.path.dirname(script_name), exist_ok=True)
    with open(script_name, 'w+') as f:
        f.write('SCRIPT_PATH=$(realpath $0)\n')
        f.write('SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")\n')
        for file in files:
            f.write(f"/bin/bash $1 $SCRIPT_DIR/yamls/{file} $2\n")


for (arch, n_threads), packs in files.items():
    files_for_parts = [[] for x in range(dlmc_parts)]

    # Create scripts for each pack
    for (pack_name, part), files in packs.items():
        gen_run_script(files, f"{GENERATED_DIR}/{arch}/{pack_name}_part{part}.sh")
        files_for_parts[part-1] += files

    for i in range(1, dlmc_parts+1):
        gen_run_script(files_for_parts[i-1], f"{GENERATED_DIR}/{arch}/all_part{i}.sh")



# for pack_name in pack_names:
#     gen_cluster_scripts(sub_dir, run_all_script_name + "_" + pack_name)
# gen_cluster_scripts(sub_dir, run_all_script_name)
