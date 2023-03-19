#!/bin/bash

cd ..
rsync -rltv --progress --exclude-from=spmm-nano-bench/rsync_exclude.txt spmm-nano-bench $1@niagara.computecanada.ca:/scratch/m/mmehride/$1/work_private
cd spmm-nano-bench/

