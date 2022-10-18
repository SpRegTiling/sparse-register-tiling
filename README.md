# How to run benchmark on AVX2

## generating vectorized code
from `spmm-nano-kernels`, run the following:
```
 python3 -m codegen.generate_ukernels
```

## Building the code
```bash
cd build
cmake -DBENCHMARK_ENABLE_GTEST_TESTS=OFF -DENABLE_AVX2=True .. 
make 
```

## Running the code
set working directory to where this project is located. 

cpp_testbed/demo/SPMM_demo -d <path to datasets> -e 
tools/experiments/example_avx512_mobilenet.yaml

```bash
cd spmm-nano-bench
./build/SPMM_demo -d where/datasets/are/located -e tools/experiments/config.
yaml 
```