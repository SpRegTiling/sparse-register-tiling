SBATCH=sbatch

if ! command -v sbatch &> /dev/null
then
    echo "sbatch could not be found, running locally"
    SBATCH=sh
fi

$SBATCH <<EOT
#!/bin/sh

#SBATCH --cpus-per-task=40
#SBATCH --export=ALL
#SBATCH --job-name="log_figure13_8"
#SBATCH --nodes=1
#SBATCH --account=def-kazem
#SBATCH --output="log_figure13_8.%j.%N.out"
#SBATCH -t 6:00:00
#SBATCH --constraint=cascade

singularity exec -B ../:/datasets -B ../results:/results spreg.sif python /spmm-nano-bench/sbench/SOP/sop_bench_ilp_sweep.py /spmm-nano-bench/sbench/SOP/sweep_mappings/ilp/8 8 /results/
EOT