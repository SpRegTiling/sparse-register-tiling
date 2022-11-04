#!/bin/bash

# Submit to sbatch
sbatch <<EOT
#!/bin/sh

#SBATCH --cpus-per-task=40
#SBATCH --export=ALL
#SBATCH --job-name="log_microbench"
#SBATCH --nodes=1
#SBATCH --output="log_microbench.%j.%N.out"
#SBATCH -t 0:30:00
#SBATCH --constraint=cascade
#SBATCH --account=def-kazem

singularity run --bind spmm-nano-bench:/mnt/spmm-nano-bench --bind dlmc:/mnt/datasets/dlmc -H /mnt/spmm-nano-bench spmm-nano-bench.sif python sbench/SOP/sop_microbench.py

EOT