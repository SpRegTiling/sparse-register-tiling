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
#SBATCH --job-name="double_$1 threads_$2"
#SBATCH --nodes=1
#SBATCH --account=def-kazem
#SBATCH --output="log_double_.$1.$2.%j.%N.out"
#SBATCH -t 8:30:00
#SBATCH --constraint=skylake

echo singularity exec -B ../:/datasets -B ../results:/results spreg.sif  "python /spmm-nano-bench/artifact/double/run_double.py --mtx_range $1 --threads $2; /tmp/experiment_scripts/figure_double_$1_$2.sh"
singularity exec -B ../:/datasets -B ../results:/results spreg.sif python /spmm-nano-bench/artifact/double/run_double.py --mtx_range $1 --threads $2
singularity exec -B ../:/datasets -B ../results:/results spreg.sif sh /tmp/experiment_scripts/figure_double_$1_$2.sh

EOT