#!/bin/bash

# Submit to sbatch
sbatch <<EOT
#!/bin/sh

#SBATCH --cpus-per-task=40
#SBATCH --export=ALL
#SBATCH --job-name="log_microbench"
#SBATCH --nodes=1
#SBATCH --output="log_microbench.%j.%N.out"
#SBATCH -t 2:00:00
#SBATCH --constraint=cascade

singularity run --bind dnn-spmm-bench:/dnn-spmm-bench --bind dlmc:/mnt/datasets/dlmc --bind .:/mnt/workdir --env PYTHONPATH=/dnn-spmm-bench --pwd /mnt/workdir dnn-spmm-bench.sif python /dnn-spmm-bench/spmm_benchmarks/SOP/sop_microbench.py

EOT