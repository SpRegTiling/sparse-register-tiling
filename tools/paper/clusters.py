import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

clusters = {
    "local":   ("AVX512", 8),
    "niagara": ("AVX512", 32),
    "graham":  ("AVX2", 32),
    "narval":  ("AVX2", 64),
}

HOURS = 24


def gen_cluster_scripts(sub_dir, run_all_script_name):
    for cluster in clusters.keys():
        with open(SCRIPT_DIR + f"/experiments/generated/{cluster}_{run_all_script_name}.sh", 'w+') as f:
            arch, max_thread_count = clusters[cluster]

            f.write(f'''#!/bin/bash -i 

SCRIPT_PATH=$(realpath $0)
SCRIPT_DIR=$(dirname "${{SCRIPT_PATH}}")
/bin/bash $SCRIPT_DIR/{sub_dir}/{run_all_script_name}_{arch}_{max_thread_count}_runall.sh \\
    $SCRIPT_DIR/../../../clusters/exp_{cluster}.sh {HOURS}
''')