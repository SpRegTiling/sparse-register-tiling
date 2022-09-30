#!/bin/bash

# Submit to sbatch
sbatch <<EOT
#!/bin/sh

#SBATCH --cpus-per-task=40
#SBATCH --export=ALL
#SBATCH --job-name="$EXPERIMENT_NAME"
#SBATCH --nodes=1
#SBATCH --output="info_$EXPERIMENT_NAME.%j.%N.out"
#SBATCH -t 00:15:00
#SBATCH --constraint=cascade

lscpu

EOT