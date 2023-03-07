import torch
from torch.utils.cpp_extension import load, include_paths
import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
import numpy as np
import math
import glob
import hashlib

import spmm_nano_kernels.codegen.base_ukernel_codegen as ukernel_codegen
from sbench.SOP.sop_cost_model import Acc
from sbench.SOP.sop_utils import Acc, pattern_code

kernel_cache = {}
os.makedirs('/tmp/sp_reg_python/_C/generated/', exist_ok=True)

cflags = ['-DUSE_JIT=1', '-std=c++17', '-Ofast', '-ffast-math',
          '-march=native', '-mtune=intel', '-mavx512pf',
          '-ffp-contract=fast']

include_dirs = [
    f'{SCRIPT_DIR}/../../spmm_nano_kernels/include',
    f'{SCRIPT_DIR}/../../spmm_nano_kernels/generated/AVX512/include',
    f'{SCRIPT_DIR}/../../third_party',
    f'{SCRIPT_DIR}/../../spmm_nano_kernels/third_party/version2/vectorclass',
    f'{SCRIPT_DIR}/../../spmm_nano_kernels/third_party/rte',
    f'/tmp/sp_reg_python/_C/generated',
    f'/tmp/sp_reg_python/_C/generated/AVX512/include',
]

sop_sources = glob.glob(f'{SCRIPT_DIR}/../../SOP/src/**/*.cpp', recursive=True) + \
        glob.glob(f'/tmp/sp_reg_python/_C/generated/**/*.cpp', recursive=True)

csr = load(name=f'csr',
           sources=[f'{SCRIPT_DIR}/_C/pytorch_csr_wrapper.cpp'],
           extra_include_paths=include_dirs, extra_cflags=cflags, extra_cuda_cflags=cflags,
           build_directory=f'/tmp/sp_reg_python/_C/generated/')

class SOPModule:
    def __init__(self, acc: Acc, patterns, pattern_mapping, c_module, kernel_id):
        self.kernel_id = kernel_id
        self.acc = acc
        self.patterns = patterns
        self.pattern_mapping = pattern_mapping
        self.c_module = c_module
        self._SOPTile = getattr(c_module, f'SOPTile_{kernel_id}')

    def make_sop_tile(self, tile, col_offset=0):
        num_panels = math.ceil(tile.shape[0] / self.acc.M)

        def convert_panel_to_codes(A):
            _pat_codes = []
            _col_indices = []

            for i, row in enumerate(A):
                codes = self.pattern_mapping(pattern_code(row))
                if type(codes) == int:
                    _pat_codes.append(codes)
                    _col_indices.append(i)
                elif type(codes) == list:
                    for code in codes:
                        _pat_codes.append(code)
                        _col_indices.append(i)

            return torch.as_tensor(np.array(_pat_codes), dtype=torch.int), \
                   torch.as_tensor(np.array(_col_indices), dtype=torch.int)

        sop_tile = self.SOPTile(*tile.shape, num_panels)
        for panel_id in range(num_panels):
            panel = tile[panel_id*self.acc.M: (panel_id + 1)*self.acc.M, :].clone().t().contiguous()
            pat_codes, col_indices = convert_panel_to_codes(panel)
            sop_tile.pack_panel(panel_id, col_offset, pat_codes, col_indices, panel)

        sop_tile.pack_tile()
        return sop_tile

    def SOPTile(self, rows: int, cols: int, num_panels: int):
        return self._SOPTile(rows, cols, num_panels)

    def execute_tile(self, tile, B: torch.Tensor, num_runs=1024, N_c=None):
        N_c = B.shape[-1] if N_c is None else N_c
        return self.c_module.execute_tile(N_c, tile, B, num_runs)

    def executor(self, csr: torch.Tensor, B: torch.Tensor, num_runs=1024, N_c=None):
        N_c = B.shape[-1] if N_c is None else N_c
        return self.c_module.executor(N_c, csr, B, num_runs)


def make_sop_module(acc: Acc, patterns, pattern_mapping, regen=False) -> SOPModule:
    global kernel_cache
    assert 0 not in patterns, "Zero is not a valid pattern"

    pattern_check_sum = sum([0]) + len(patterns)
    codegen = ukernel_codegen.UKernelCodegenBase(Mr=acc.M, nanokernels=list(patterns),
                                                 output_root='/tmp/sp_reg_python/_C/generated/')

    microkernel_typename = codegen.typename('float', 'AVX512', 512, acc.N)

    print(microkernel_typename)

    if not regen and microkernel_typename in kernel_cache:
        c_module = kernel_cache[microkernel_typename]
    else:
        if regen or not os.path.exists(f'/tmp/sp_reg_python/_C/generated/pytorch_wrapper_{microkernel_typename}.cpp'):
            print(f'Generating: tmp/sp_reg_python/_C/generated/{microkernel_typename}.h')
            print(f'Generating: tmp/sp_reg_python/_C/generated/pytorch_wrapper_{microkernel_typename}.cpp')
            codegen.gen_header(acc.N, 'AVX512', 512)
            header_location = f'{codegen.nanokernel_hash}/{microkernel_typename}_datatransform_true.h'

            with open(f'{SCRIPT_DIR}/_C/pytorch_wrapper_intrin.cpp.template') as f:
                template = f.read()
                template = template \
                    .replace('template_MICROKERNEL_HEADER', header_location) \
                    .replace("template_KERNEL_ID", microkernel_typename) \
                    .replace("template_MAX_ACC_WIDTH", str(acc.N)) \
                    .replace("template_PANEL_HIEGHT", str(acc.M)) \
                    .replace("template_NAMESPACE", f'sop_bench_{microkernel_typename}') \
                    .replace("template_MICROKERNEL_TYPENAME", microkernel_typename)

            with open(f'/tmp/sp_reg_python/_C/generated/pytorch_wrapper_{microkernel_typename}.cpp', 'w+') as f:
                f.write(template)

        c_module = load(name=f'sop_{microkernel_typename}',
                        sources=[f'/tmp/sp_reg_python/_C/generated/pytorch_wrapper_{microkernel_typename}.cpp'] + sop_sources,
                        extra_include_paths=include_dirs, extra_cflags=cflags, extra_cuda_cflags=cflags,
                        build_directory=f'/tmp/sp_reg_python/_C/generated/')

        kernel_cache[microkernel_typename] = c_module

    return SOPModule(acc, patterns, pattern_mapping, c_module, microkernel_typename)




