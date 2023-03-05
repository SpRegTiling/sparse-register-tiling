#!/bin/bash

cd ..
rsync -rltv --progress --exclude-from=spmm-nano-bench/rsync_exclude.txt spmm-nano-bench $1@niagara.computecanada.ca:/scratch/m/mmehride/$1/sp_reg_artifact
cd spmm-nano-bench/

