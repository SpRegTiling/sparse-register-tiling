#!/bin/bash

# Submit to sbatch
sbatch <<EOT
#!/bin/sh

#SBATCH --cpus-per-task=40
#SBATCH --export=ALL
#SBATCH --job-name="log_sop_bench"
#SBATCH --nodes=1
#SBATCH --output="log_sop_bench.%j.%N.out"
#SBATCH -t 10:00:00
#SBATCH --constraint=cascade

singularity run --bind dnn-spmm-bench:/dnn-spmm-bench --bind dlmc:/mnt/datasets/dlmc --env PYTHONPATH=/dnn-spmm-bench dnn-spmm-bench.sif python /dnn-spmm-bench/spmm_benchmarks/SOP/sop_bench.py

EOT