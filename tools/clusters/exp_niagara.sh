#!/bin/bash

EXPERIMENT_NAME=$(basename -s .yaml $1)
SCRIPT_PATH=$(realpath $0)
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
BUILD_PATH=$SCRIPT_DIR/../../build
SOURCE_PATH=$SCRIPT_DIR/../..
DATASET_DIR_PATH=$SCRIPT_DIR/../../..
SPMM_BIN_PATH=$BUILD_PATH/cpp_testbed/demo/SPMM_demo

module load NiaEnv/2019b
module load cmake/3.17.3
module load gcc
module load metis/5.1.0
module load papi
export MKLROOT=/scinet/intel/oneapi/2021u4/mkl/2021.4.0/

# Configure before we submit so that access the internet
cmake -S $SOURCE_PATH -B $BUILD_PATH -DCMAKE_BUILD_TYPE=Release -DENABLE_AVX512=True ..

echo CLEAN ${CLEAN}
if [ "${CLEAN}" ]; then
  make -C $BUILD_PATH clean
fi

make -C $BUILD_PATH -j 20 SPMM_demo

echo ""
echo "Submitting using:"
echo "   EXPERIMENT=$1"
echo "   BUILD_PATH=$BUILD_PATH"
echo "   SOURCE_PATH=$SOURCE_PATH"
echo "   DATASET_DIR_PATH=$DATASET_DIR_PATH"
echo "   SPMM_BIN_PATH=$SPMM_BIN_PATH"
echo ""

echo ${INTERACTIVE}
if [ "${INTERACTIVE}" ]; then
    echo "INTERACTIVE set, skipping submit and running commands here"

    $SPMM_BIN_PATH -e $1 -d $DATASET_DIR_PATH

    exit 0
fi

# Submit to sbatch
sbatch <<EOT
#!/bin/sh

#SBATCH --cpus-per-task=40
#SBATCH --export=ALL
#SBATCH --job-name="$EXPERIMENT_NAME"
#SBATCH --nodes=1
#SBATCH --account=def-kazem
#SBATCH --output="log_$EXPERIMENT_NAME.%j.%N.out"
#SBATCH -t $2:00:00
#SBATCH --constraint=cascade

module load NiaEnv/2019b
module load cmake/3.17.3
module load gcc
module load metis/5.1.0
module load papi

export MKL_ENABLE_INSTRUCTIONS=AVX512
export OMP_PROC_BIND=true
$SPMM_BIN_PATH -e $1 -d $DATASET_DIR_PATH

EOT