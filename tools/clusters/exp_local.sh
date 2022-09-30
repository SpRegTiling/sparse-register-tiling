#!/bin/bash -i 

EXPERIMENT_NAME=$(basename -s .yaml $1)
SCRIPT_PATH=$(realpath $0)
SCRIPT_DIR=$(dirname "${SCRIPT_PATH}")
BUILD_PATH=$SCRIPT_DIR/../../build
SOURCE_PATH=$SCRIPT_DIR/../..
DATASET_DIR_PATH=$SCRIPT_DIR/../../..
SPMM_BIN_PATH=$BUILD_PATH/cpp_testbed/demo/SPMM_demo


# Configure before we submit so that access the internet
cmake -S $SOURCE_PATH -B $BUILD_PATH -DCMAKE_BUILD_TYPE=Release -DENABLE_AVX512=True ..

echo CLEAN ${CLEAN}
if [ "${CLEAN}" ]; then
  make -C $BUILD_PATH clean
fi

make -C $BUILD_PATH -j 20 SPMM_demo

export OMP_PROC_BIND=true
$SPMM_BIN_PATH -e $1 -d $DATASET_DIR_PATH

exit 0
