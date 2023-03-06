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
#SBATCH --job-name="figure7_to_9_$1 threads_$2"
#SBATCH --nodes=1
#SBATCH --account=def-kazem
#SBATCH --output="log_figure7_to_9.$1.$2.%j.%N.out"
#SBATCH -t 2:00:00
#SBATCH --constraint=cascade

singularity exec -B ../plots:/plots -B ../results:/results spreg.sif python /spmm-nano-bench/artifact/figure7_to_9/box_plots.py

EOT



