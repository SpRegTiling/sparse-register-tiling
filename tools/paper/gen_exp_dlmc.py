from spmm_benchmarks.loaders.dlmc import DLMCPathIterator

from tools.paper.configs import nano, csb
from tools.paper.clusters import gen_cluster_scripts

from collections.abc import Iterable
import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
import yaml

sub_dir = 'dlmc'
run_all_script_name = 'all_dlmc'


def flatten(coll):
    for i in coll:
        if isinstance(i, Iterable) and not isinstance(i, str):
            for subc in flatten(i):
                yield subc
        else:
            yield i


with open(SCRIPT_DIR + "/../filelists/all_dlmc_part1.txt", 'w+') as f:
    for path in list(DLMCPathIterator())[:1300]:
        f.write("dlmc" + path.split("dlmc")[-1] + "\n")

with open(SCRIPT_DIR + "/../filelists/all_dlmc_part2.txt", 'w+') as f:
    for path in list(DLMCPathIterator())[1300:2600]:
        f.write("dlmc" + path.split("dlmc")[-1] + "\n")

with open(SCRIPT_DIR + "/../filelists/all_dlmc_part3.txt", 'w+') as f:
    for path in list(DLMCPathIterator())[2600:]:
        f.write("dlmc" + path.split("dlmc")[-1] + "\n")


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
        "filelist": f'../../../../../filelists/{filelist_name}.txt',
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

    experiment_file = f'{filelist_name}_{arch}{suffix}.yaml'
    dir = f'/experiments/generated/dlmc/{arch}/'
    os.makedirs(SCRIPT_DIR + dir, exist_ok=True)

    with open(SCRIPT_DIR + dir + experiment_file, 'w+') as f:
        yaml.dump({
            "options": options,
            "tuning": {
                "parameter_grids": tuning_grids,
            },
            "matrices": matrices,
            "methods": baseline_methods + test_methods
        }, f)

    return experiment_file


# All-DLMC Experiments
for arch in ["AVX2", "AVX512"]:
    for max_thread_count in [8, 32, 64]:
        n_threads = {
            8: [1, 8],
            32: [1, 16, 32],
            64: [1, 16, 32, 64],
        }

        test_methods_nano4_aspt = [
            {
                "name": "ASpT",
                "method_id": "aspt",
                "options": {
                    "vec_width": "not-supported"
                }
            },

            nano(arch, 4, "identity"),
            nano(arch, 4, "identity", load_balance=True),
            nano(arch, 4, "identity", load_balance=True, tlb_comp=64),
            nano(arch, 4, "identity", load_balance=True, sparse_a=True),
            nano(arch, 4, "identity", load_balance=True, sparse_a=True, tlb_comp=48),
            nano(arch, 4, "identity", load_balance=True, sparse_a=True, tlb_comp=64),
            nano(arch, 4, "identity", load_balance=True, sparse_a=True, tlb_comp=128),
            nano(arch, 4, "identity", load_balance=True, sparse_a=True, tlb_comp=64, beta=1.5),
            nano(arch, 4, "identity", load_balance=True, sparse_a=True, tlb_comp=64, beta=2.0),
            nano(arch, 4, "identity", load_balance=True, sparse_a=True, tlb_comp=128, beta=2.0),
            nano(arch, 4, "identity", load_balance=True, sparse_a=True, tlb_comp=64, beta=3.0),
            nano(arch, 4, "orig", load_balance=True),
            nano(arch, 4, "orig", load_balance=True, sparse_a=True, tlb_comp=64),
        ]

        test_methods_nano8 = [
            nano(arch, 8, "orig"),
            nano(arch, 8, "orig", load_balance=True),
            nano(arch, 8, "orig", load_balance=True, tlb_comp=64),
            nano(arch, 8, "orig", load_balance=True, sparse_a=True, tlb_comp=48),
            nano(arch, 8, "orig", load_balance=True, sparse_a=True, tlb_comp=64),
            nano(arch, 8, "orig", load_balance=True, sparse_a=True, tlb_comp=128),
            nano(arch, 8, "alt",  load_balance=True, sparse_a=True, tlb_comp=64),
        ]

        test_methods_csb = [
            csb(arch, "CSR", 32),
            csb(arch, "CSR", 32, sparse_a=True, tlb_comp=64),
            csb(arch, "CSR", 64),
            csb(arch, "CSR", 64, sparse_a=True, tlb_comp=64),
        ]

        test_methods_nano_tuned = [
            nano(arch, 4, "identity", load_balance=True, tune="SOP4"),
            nano(arch, 4, "orig", load_balance=True, tune="SOP4"),
            nano(arch, 8, "orig", load_balance=True, tune="SOP4"),
            nano(arch, 8, "alt", load_balance=True, tune="SOP4")
        ]

        files = [
            [
                gen_dlmc_bench_exp(arch, test_methods_nano4_aspt, filelist,
                                   b_cols=[32, 128, 256],
                                   num_threads=n_threads[max_thread_count],
                                   suffix=f"nano4_all_threads_small_bcols_{max_thread_count}"),
                gen_dlmc_bench_exp(arch, test_methods_nano4_aspt, filelist,
                                   b_cols=[512, 1024],
                                   num_threads=n_threads[max_thread_count],
                                   suffix=f"nano4_all_threads_large_bcols_{max_thread_count}"),
                gen_dlmc_bench_exp(arch, test_methods_nano8, filelist,
                                   b_cols=[32, 128, 256],
                                   num_threads=n_threads[max_thread_count],
                                   suffix=f"nano8_all_threads_small_bcols_{max_thread_count}"),
                gen_dlmc_bench_exp(arch, test_methods_csb, filelist,
                                   b_cols=[32, 128, 128],
                                   num_threads=n_threads[max_thread_count],
                                   suffix=f"csb_all_threads_small_bcols_{max_thread_count}"),
                gen_dlmc_bench_exp(arch, test_methods_csb, filelist,
                                   b_cols=[512, 1024],
                                   num_threads=n_threads[max_thread_count],
                                   suffix=f"csb_all_threads_large_bcols_{max_thread_count}"),
                gen_dlmc_bench_exp(arch, test_methods_nano_tuned, filelist,
                                   b_cols=[32, 128, 256],
                                   num_threads=n_threads[max_thread_count],
                                   suffix=f"nano_tuned_all_threads_small_bcols_{max_thread_count}"),
           ] for filelist in ["all_dlmc_part1", "all_dlmc_part2", "all_dlmc_part3"]
        ]

        files = flatten(files)
        run_all_script = SCRIPT_DIR + \
             f"/experiments/generated/{sub_dir}/{run_all_script_name}_{arch}_{max_thread_count}_runall.sh"

        print("Generating runall script: ", run_all_script)
        with open(run_all_script, 'w+') as f:
            f.write('SCRIPT_PATH=$(realpath $0)\n')
            f.write('SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")\n')
            for file in files:
                f.write(f"/bin/bash $1 $SCRIPT_DIR/{arch}/{file} $2\n")

gen_cluster_scripts(sub_dir, run_all_script_name)
