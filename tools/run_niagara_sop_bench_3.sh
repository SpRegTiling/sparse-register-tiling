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
#SBATCH --account=def-kazem

singularity run --bind spmm-nano-bench:/mnt/spmm-nano-bench \
 --bind dlmc:/mnt/datasets/dlmc \
 -H /mnt/spmm-nano-bench spmm-nano-bench.sif \
 python sbench/SOP/sop_bench_ilp_sweep.py sbench/SOP/sweep_mappings/ilp/4/ 4

singularity run --bind spmm-nano-bench:/mnt/spmm-nano-bench \
 --bind dlmc:/mnt/datasets/dlmc \
 -H /mnt/spmm-nano-bench spmm-nano-bench.sif \
 python sbench/SOP/sop_bench_ilp_sweep.py sbench/SOP/sweep_mappings/ga/4/ 4

singularity run --bind spmm-nano-bench:/mnt/spmm-nano-bench \
 --bind dlmc:/mnt/datasets/dlmc \
 -H /mnt/spmm-nano-bench spmm-nano-bench.sif \
 python sbench/SOP/sop_bench_ilp_sweep.py sbench/SOP/sweep_mappings/ga/6/ 6

singularity run --bind spmm-nano-bench:/mnt/spmm-nano-bench \
 --bind dlmc:/mnt/datasets/dlmc \
 -H /mnt/spmm-nano-bench spmm-nano-bench.sif \
 python sbench/SOP/sop_bench_ilp_sweep.py sbench/SOP/sweep_mappings/ga/8/ 8
EOT