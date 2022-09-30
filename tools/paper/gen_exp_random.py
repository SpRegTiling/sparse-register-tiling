from spmm_benchmarks.loaders.dlmc import DLMCPathIterator
from tools.paper.configs import nano, csb
import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
import yaml


with open(SCRIPT_DIR + "/filelists/all_dlmc_part1.txt", 'w+') as f:
    for path in list(DLMCPathIterator())[:1300]:
        f.write("dlmc" + path.split("dlmc")[-1] + "\n")

with open(SCRIPT_DIR + "/filelists/all_dlmc_part2.txt", 'w+') as f:
    for path in list(DLMCPathIterator())[1300:2600]:
        f.write("dlmc" + path.split("dlmc")[-1] + "\n")

with open(SCRIPT_DIR + "/filelists/all_dlmc_part3.txt", 'w+') as f:
    for path in list(DLMCPathIterator())[2600:]:
        f.write("dlmc" + path.split("dlmc")[-1] + "\n")


def gen_dlmc_bench_exp(vec_width, filelist, b_cols, num_threads, suffix = ""):
    filelist_name = filelist.split("/")[-1].replace(".txt", "")

    if suffix != "" and suffix[0] != "_":
        suffix = "_" + suffix

    options = {
        "profile": False,
        "scalar_type": "float",
        "b_cols": b_cols,
        "n_threads": num_threads,
        "output_file": f"results/{filelist_name}_{vec_width}{suffix}.csv",
        "save_tuning_results": False,
        "expand_config_parameters": [ "m_tile", "k_tile", "n_tile", "tiling_strategy", "sparse_a", "beta_10x" ]
    }

    matrices = {
        "filelist": f'../../../tools/filelists/{filelist_name}.txt',
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

    baseline_methods = [
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
    ]

    test_methods = [
        nano(4, "identity"),
        nano(4, "orig", load_balance=True),
        nano(4, "identity", load_balance=True),
        nano(4, "identity", load_balance=True, sparse_a=True),
        nano(4, "identity", load_balance=True, sparse_a=True, tlb_comp=True),
        nano(4, "identity", load_balance=True, sparse_a=True, tlb_comp=True, beta=1.5),
        nano(4, "identity", load_balance=True, sparse_a=True, tlb_comp=True, beta=2.0),
        nano(4, "identity", load_balance=True, sparse_a=True, tlb_comp=True, beta=3.0),
        nano(4, "orig", load_balance=True, sparse_a=True, tlb_comp=True),
        nano(8, "orig"),
        nano(8, "orig", load_balance=True),
        nano(8, "orig", load_balance=True, sparse_a=True),
        nano(8, "orig", load_balance=True, sparse_a=True, tlb_comp=True),
        nano(8, "alt", load_balance=True, sparse_a=True, tlb_comp=True),
    ]

    experiment_file = f'{filelist_name}_{vec_width}{suffix}.yaml'
    with open(SCRIPT_DIR + f"/experiments/generated/{experiment_file}", 'w+') as f:
        yaml.dump({
            "options": options,
            "tuning": {
                "parameter_grids": tuning_grids,
            },
            "matrices": matrices,
            "methods": baseline_methods + test_methods
        }, f)

    return experiment_file



def gen_random_sweep_exp(vec_width, filelist,
                         b_cols = [256, 512],
                         num_threads = [1, 16],
                         suffix = ""):
    filelist_name = filelist.split("/")[-1].replace(".txt", "")

    if suffix != "" and suffix[0] != "_":
        suffix = "_" + suffix

    options = {
        "profile": True,
        "scalar_type": "float",
        "b_cols": b_cols,
        "n_threads": num_threads,
        "output_file": f"results/{filelist_name}_{vec_width}{suffix}.csv",
        "save_tuning_results": False,
        "expand_config_parameters": [ "m_tile", "k_tile", "n_tile", "tiling_strategy", "sparse_a", "beta_10x" ]
    }

    matrices = {
        "filelist": f'../../../tools/filelists/{filelist_name}.txt',
    }

    baseline_methods = [
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
        },
        {
            "name": "ASpT",
            "method_id": "aspt",
            "options": {
                "vec_width": "not-supported"
            }
        }
    ]

    test_methods = [
        csb("CSR", 32),
        csb("CSR", 32, sparse_a=True, tlb_comp=True),
        csb("CSC", 32),
        csb("CSC", 32, sparse_a=True, tlb_comp=True),
        csb("CSR", 64),
        csb("CSR", 64, sparse_a=True, tlb_comp=True),
        csb("CSC", 64),
        csb("CSC", 64, sparse_a=True, tlb_comp=True),
        nano(4, "identity"),
        nano(4, "identity", sparse_a=True, tlb_comp=True),
        nano(4, "orig"),
        nano(4, "orig", sparse_a=True, tlb_comp=True),
        nano(4, "identity", load_balance=True, sparse_a=True, tlb_comp=True),
        nano(4, "identity", load_balance=True, sparse_a=True, tlb_comp=True, beta=3.0),
        nano(8, "orig"),
        nano(8, "orig", load_balance=True),
        nano(8, "orig", load_balance=True, sparse_a=True),
        nano(8, "orig", load_balance=True, sparse_a=True, tlb_comp=True),
        nano(8, "alt", load_balance=True, sparse_a=True, tlb_comp=True),
    ]

    experiment_file = f'{filelist_name}_{vec_width}{suffix}.yaml'
    with open(SCRIPT_DIR + f"/experiments/generated/{experiment_file}", 'w+') as f:
        yaml.dump({
            "options": options,
            "matrices": matrices,
            "methods": baseline_methods + test_methods
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
for vec_width in [256, 512]:
    for max_thread_count in [32, 64]:
        n_threads = {
            32: [1, 16, 32],
            64: [1, 16, 32, 64],
        }

        files = [
            [gen_dlmc_bench_exp(vec_width, filelist, b_cols=[32, 128, 256], num_threads=n_threads[max_thread_count],
                                suffix=f"all_threads_small_bcols_{max_thread_count}"),
             gen_dlmc_bench_exp(vec_width, filelist, b_cols=[512, 1024], num_threads=n_threads[max_thread_count],
                                suffix=f"all_threads_large_bcols_{max_thread_count}")]

            for filelist in ["all_dlmc_part1", "all_dlmc_part2", "all_dlmc_part3"]
        ]


        files = flatten(files)
        with open(SCRIPT_DIR + f"/experiments/generated/all_dlmc_{vec_width}_{max_thread_count}_runall.sh", 'w+') as f:
            for file in files:
                f.write(f"/bin/bash dnn-spmm-bench/tools/$1 "
                        f"dnn-spmm-bench/tools/experiments/generated/{file}\n")


# Random Sweep Experiments
for vec_width in [256, 512]:
    for max_thread_count in [32]:
        n_threads = {32: [1, 16, 32]}

        files = [
            [gen_random_sweep_exp(vec_width, filelist, b_cols=[128, 256], num_threads=n_threads[max_thread_count],
                                  suffix=f"random_sweep_{max_thread_count}")]

            for filelist in ["random_sweep"]
        ]

        files = flatten(files)
        with open(SCRIPT_DIR + f"/experiments/generated/random_sweep_{vec_width}_{max_thread_count}_runall.sh", 'w+') as f:
            for file in files:
                f.write(f"/bin/bash dnn-spmm-bench/tools/$1 "
                        f"dnn-spmm-bench/tools/experiments/generated/{file}\n")