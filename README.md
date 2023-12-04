> **Warning** We intend on creating a more accessible api for integrating the SpMM kernels into other projects and recommend waiting for that (over working with this codebase) if possible.

# Building and running from source

ensure your machine has avx512vl
```
lscpu | grep avx512vl
```

clone the repo
```
git clone https://github.com/SpRegTiling/sparse-register-tiling.git --recurse-submodules
cd sparse-register-tiling
```

If cloning submodules fails, e.g. due to github key setup, you may clone the the submodule manually by:
```
git clone https://github.com/LucasWilkinson/spmm-nano-kernels.git spmm_nano_kernels
```


## Download DLMC

Download the dlmc dataset, this will download it to `../dlmc`

```
sh download_dlmc.sh
```

## Install python dependencies

```
pip3 install torch
pip3 install -e .
pip3 install pandas  \
     matplotlib \
     scipy \
     pyyaml
export PYTHONPATH=$(pwd):$PYTHONPATH
```

## Install MKL 2021.4.0
```
wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB && \
    add-apt-repository -y "deb https://apt.repos.intel.com/oneapi all main" && \
    apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB && \
    rm GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB && \
    apt-get update -y && \
    apt-get install -y intel-oneapi-mkl-devel-2021.4.0
```

## Generating executor/scheduler pairs (used in the paper)
from `spmm_nano_kernels`, run the following:
```
cd spmm_nano_kernels
python3 -m codegen.generate_ukernels
cd ..
```


## Building the demo code
```
mkdir release-build
cmake -Brelease-build -DCMAKE_BUILD_TYPE=Release -DENABLE_AVX512=True .
make -j 16 -Crelease-build SPMM_demo
```

## Benchmarking a matrix

```
python3 run_matrix.py -m ../dlmc/transformer/magnitude_pruning/0.8/body_decoder_layer_0_ffn_conv1_fully_connected.smtx -t 8 -b 512 -o results.csv -d
```
see:
```
python3 run_matrix.py --help
```
for more details

# Code overview

- `cpp_testbed/demo/SPMM_demo.cpp`: benchmarking code (best to use the `python3 artifact/run_matrix.py` wrapper)
- `spmm_nano_kernels/codegen/generate_ukernels.py`: entry point for generating scheduler/executor pairs
- `spmm_nano_kernels/codegen/base_ukernel_codegen.py`: code for generating mini-register tiles, this code in conjunction with `spmm_nano_kernels/include/Executor.h` forms the executor code
- `spmm_nano_kernels/include/MicroKernelPacker.h`: templated scheduler + data-compressor code
- `spmm_nano_kernels/include/MatMulSpecialized.h`: wrapper that contains a scheduler/executor pair
- `sbench/ilp_pad/nano_solver.py` contains the code for the mathematical model
