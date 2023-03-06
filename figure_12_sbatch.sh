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
#SBATCH --job-name="log_figure12"
#SBATCH --nodes=1
#SBATCH --account=def-kazem
#SBATCH --output="log_figure12.%j.%N.out"
#SBATCH -t 2:00:00
#SBATCH --constraint=cascade

echo "singularity exec -B ../:/datasets -B ../results:/results spreg.sif python /spmm-nano-bench/artifact/figure12/run.py"
singularity exec -B ../:/datasets -B ../results:/results spreg.sif python /spmm-nano-bench/artifact/figure12/run.py
singularity exec -B ../:/datasets -B ../results:/results spreg.sif sh /tmp/experiment_scripts/figure12.sh
EOT