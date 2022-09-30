import gmpy

from tools.codegen.codegen_utils import *
from SOP.codegen.sop_codegen import gen_for_vec_height

SUPPORTED_PATTERNS_16 = [
    1 << i for i in range(16)
] + [

   0b0100010001000100, 0b1010101010101010,
   0b11000011, 0b00111100, 0b00001111, 0b11110000,
   0b11111100, 0b11110011, 0b11001111, 0b00111111,
   0b11111111
]


SUPPORTED_PATTERNS_8 = [
    1 << i for i in range(8) #3 << i for i in range(0, 8, 2)
] + [
    0b01010101, 0b10101010,
    0b11000011, 0b00111100, 0b00001111, 0b11110000,
    0b11111100, 0b11110011, 0b11001111, 0b00111111,
    0b11111111
]


# SUPPORTED_PATTERNS_4 = [
#     0b0011, 0b1100, 0b0110, 0b0110, 0b1111
# ]
SUPPORTED_PATTERNS_4 = [
    0b00000001, 0b00000010, 0b00000100, 0b00001000,
    0b00001110, 0b00001101, 0b00001011, 0b00000111,
    0b00001111,
]

import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
gen_for_vec_height('', [8, 2], SUPPORTED_PATTERNS_8,
                   output_path=f'{SCRIPT_DIR}/../../cpp_testbed/demo/SpMM_methods/sop_executor_8_2.h')
gen_for_vec_height('', [4, 4], SUPPORTED_PATTERNS_4,
                   output_path=f'{SCRIPT_DIR}/../../cpp_testbed/demo/SpMM_methods/sop_executor_4_4.h')

# ZERO_PATTERN_ID = 1 << 16 - 1
#
# FILE_HEADER = """
# #pragma once
#
# #include <spmm.h>
#
# struct __attribute__((__packed__)) PatternDesc {
#     uint16_t m_pattern;
#     uint16_t m_count;
#
#     PatternDesc(uint16_t pattern, uint16_t count): m_pattern(pattern), m_count(count) {}
#
#     PatternDesc() { m_pattern = 0; m_count = 0; }
#
#     inline uint16_t pattern() const { return m_pattern; };
#     inline uint16_t count() const { return m_count; };
# };
#
# struct RowPanelPatternStorage {
#     float* values         = nullptr;
#     int*   col_indices    = nullptr;
#     int*   pattern_counts = nullptr;
#
#     int num_nnz = 0;
#     int num_patterns = 0;
#     int num_col_indices = 0;
#
#     void free() {
#         delete[] values;
#         delete[] col_indices;
#         delete[] pattern_counts;
#
#         values = nullptr;
#         col_indices = nullptr;
#         pattern_counts = nullptr;
#     }
# };
#
#
# template<typename _Vec, int _panel_height>
# struct PatExecutor {
#     using Vec = _Vec;
#     using Scalar = typename Vec::Scalar;
#
#     static uint16_t  encode_pattern(uint16_t pattern);
#     static uint16_t  decode_pattern(uint16_t pattern);
#     static uint16_t  pat_popcount(uint16_t pattern);
#
#     static int       sop_accumulator_width_in_vecs();
#     static int       number_of_patterns();
#     static const uint16_t* supported_patterns();
#
#     static int       panel_height();
#
#     __inline __attribute__((__always_inline__)) static void _pat_executor(
#         int n_vecs,
#         uint32_t m, uint32_t k, uint32_t n,
#         struct RowPanelPatternStorage panel_desc,
#         const Scalar *__restrict__ B,
#         Scalar *__restrict__ C,
#         const bool load_c);
#
# };
#
# """ + f'''
# #define ZERO_PATTERN_ID {ZERO_PATTERN_ID}
# '''
#
#
# def gen_for_vec_height(f, acc_dims, m_supported_patterns, vec_height):
#     m_supported_patterns = sorted(m_supported_patterns, key=lambda x: gmpy.popcount(x))
#     supported_patterns_list = ",\n            ".join([f'0b{x:08b}' for x in m_supported_patterns])
#
#     supported_pattern_getter = f'''
#     static const uint16_t* supported_patterns() {{
#         static uint16_t patterns[] = {{
#             {supported_patterns_list}
#         }};
#
#         return patterns;
#     }}
#     '''
#
#     supported_patterns_map_cases = "\n        ".join([
#         f'if (pattern == 0b{x:08b}) return {i};' for i, x in enumerate(m_supported_patterns)])
#
#     supported_pattern_map = f'''
#     static uint16_t encode_pattern(uint16_t pattern) {{
#         {supported_patterns_map_cases}
#         if (pattern == 0) return ZERO_PATTERN_ID;
#         std::cerr << "Unable to map unsupported pattern " <<  (int) pattern << std::endl;
#         exit(-1);
#         return 0;
#     }}
#     '''
#
#     supported_patterns_unmap_cases = "\n        ".join([
#         f'if (pattern == {i}) return 0b{x:08b};' for i, x in enumerate(m_supported_patterns)])
#
#     supported_pattern_unmap = f'''
#     static uint16_t decode_pattern(uint16_t pattern) {{
#         {supported_patterns_unmap_cases}
#         if (pattern == ZERO_PATTERN_ID) return 0;
#         std::cerr << "Unable to unmap unsupported pattern id " << (int) pattern << std::endl;
#         exit(-1);
#         return 0;
#     }}
#     '''
#
#     pattern_popcount_map_cases = "\n        ".join([
#         f'if (pattern == {i}) return {gmpy.popcount(x)};' for i, x in enumerate(m_supported_patterns)])
#
#     pattern_popcount_map = f'''
#     static uint16_t pat_popcount(uint16_t pattern) {{
#         {pattern_popcount_map_cases}
#         if (pattern == ZERO_PATTERN_ID) return 0;
#         std::cerr << "Unable to get pop count for pattern id " << (int) pattern << std::endl;
#         exit(-1);
#         return 0;
#     }}
#     '''
#
#     function_header = f'''
#     __inline __attribute__((__always_inline__)) static void _pat_executor (
#         int n_vecs,
#         int m, int k, int n,
#         struct RowPanelPatternStorage panel_desc,
#         const Scalar *__restrict__ B,
#         Scalar *__restrict__ C,
#         const bool load_c)
#     '''
#
#     PRECOMPUTE_B = False
#     PREFETCH_B = False
#     LOAD_C = True
#
#     def def_accumulator(acc_dims, load=False):
#         if load: assert True
#         return [f'VecType cVec{i}{k}(0);'
#                 for i in range(acc_dims[0])
#                 for k in range(acc_dims[1])]
#
#     def load_accumulator(acc_dims, load=False):
#         if load: assert True
#         lines = [f'Scalar* C_temp = C;']
#
#         for i in range(acc_dims[0]):
#             for k in range(acc_dims[1]):
#                 lines += [
#                    f'VecType cVec{i}{k};',
#                    f'if (load_c) cVec{i}{k}.load(C_temp + {k} * VecType::size());',
#                    f'else        cVec{i}{k} ^= cVec{i}{k}; // zero c']
#
#             lines += [f'C_temp += n;']
#         return lines
#
#     def store_accumulator(acc_dims):
#         return [f'cVec{i}{k}.store(&C[{i} * n + {k} * VecType::size()]);'
#                 for i in range(acc_dims[0])
#                 for k in range(acc_dims[1])]
#
#     def precompute_B_ptrs():
#         block = Block()
#         block += f'const float* B_ptrs[num_col_indices];'
#         block += [
#             f'#pragma ivdep',
#             f'#pragma vector nontemporal (col_indices)',
#             f'#pragma prefetch col_indices:_MM_HINT_T1',
#             f'#pragma temporal (B_ptrs)',
#             f'#pragma unroll 16',
#             f'for (int i = 0; i < num_col_indices; i ++) {{',
#             f'    B_ptrs[i] = (const typename StorageTypes::Scalar *) uintptr_t(B) + uintptr_t(col_indices[i]) * n;',
#             f'}}',
#         ]
#         return block
#
#     def increment_B_ptrs(k):
#         block = Block()
#         block += [
#             f'#pragma ivdep',
#             f'#pragma unroll 16',
#             f'for (int i = 0; i < num_col_indices; i ++) {{',
#             f'    B_ptrs[i] += {k}* VecType::size();',
#             f'}}',
#         ]
#         return block
#
#     def prefetch_B(k):
#         block = Block()
#         block += [
#             f'#pragma unroll',
#             f'for (int i = 0; i < ({k} * VecType::size() * sizeof(float)) / 64; i ++) {{',
#             f'    __builtin_prefetch(((uint8_t*) B_curr) + 64 * i, 0, 3);',
#             f'}}',
#         ]
#         return block
#
#     def prefetch_B_next(k):
#         block = Block()
#         block += [
#             f'#pragma unroll',
#             f'for (int i = 0; i < ({k} * VecType::size() * sizeof(float)) / 64; i ++) {{',
#             f'    __builtin_prefetch(((uint8_t*) B_next) + 64 * i, 0, 3);',
#             f'}}',
#         ]
#         return block
#
#     def prefetch_B_next_next(k):
#         block = Block()
#         block += [
#             f'#pragma unroll',
#             f'for (int i = 0; i < ({k} * VecType::size() * sizeof(float)) / 64; i ++) {{',
#             f'    __builtin_prefetch(((uint8_t*) B_next_next) + 64 * i, 0, 3);',
#             f'}}',
#         ]
#         return block
#
#     def increment_B_ptr(k):
#         return f'B += {k} * VecType::size();'
#
#     def increment_C_ptr(k):
#         return f'C += {k} * VecType::size();'
#
#     def gen_pattern_case(acc_dims, id, pat):
#         case_body = Block()
#
#         count_loop = ForLoop(f'int pat_count = pattern_counts[{id}]', 'pat_count > 0', 'pat_count--')
#         count_loop += f'VecType aVec;'
#
#         if PRECOMPUTE_B:
#             for k in range(acc_dims[1]):
#                 count_loop += f'VecType bVec{k}; bVec{k}.load(B_ptrs[c_idx] + {k} * VecType::size());'
#         else:
#             for k in range(acc_dims[1]):
#                 count_loop += f'VecType bVec{k}; bVec{k}.load(B_curr + {k} * VecType::size());'
#
#         pat_tmp = pat
#         idx = 0
#         while pat_tmp:
#             if pat_tmp & 1:
#                 count_loop += f'aVec = VecType(*curr_value_ptr); curr_value_ptr++;'
#                 for k in range(acc_dims[1]):
#                     count_loop += f'cVec{idx}{k} = mul_add(aVec, bVec{k}, cVec{idx}{k});'
#
#             pat_tmp >>= 1
#             idx += 1
#
#         if PREFETCH_B:
#             count_loop += f'B_curr = B_next;'
#             count_loop += f'B_next = B_next_next;'
#             count_loop += f'B_next_next = col_indices[(++c_idx) + 2] * n + B;'
#             count_loop += prefetch_B_next_next(acc_dims[1])
#         else:
#             count_loop += f'B_curr = col_indices[++c_idx] * n + B;'
#
#         #count_loop += f'_mm_prefetch(B_temp, _MM_HINT_T0);'
#
#         case_body += count_loop
#         #case_body += 'break;'
#
#         # case_body += f'pattern = pattern_descs[pat_idx].pattern();'
#         # case_body += f'pat_count = pattern_descs[pat_idx++].count();'
#         return case_body
#
#     function_body = Block(add_braces=True)
#     function_body += f'if (panel_desc.num_patterns <= 0) return;'
#     function_body += f'float* values = panel_desc.values;'
#     function_body += f'uint32_t * col_indices = (uint32_t*) panel_desc.col_indices;'
#     function_body += f'int* pattern_counts = panel_desc.pattern_counts;'
#     function_body += f'int num_patterns = panel_desc.num_patterns;'
#     function_body += f'int num_col_indices = panel_desc.num_col_indices;'
#
#     if PRECOMPUTE_B:
#         function_body += precompute_B_ptrs()
#
#     nvecs_loops = ForLoop('int n_vec = 0', 'n_vec < n_vecs', f'n_vec += {acc_dims[1]}')
#
#     if LOAD_C:
#         nvecs_loops += load_accumulator(acc_dims)
#     else:
#         nvecs_loops += def_accumulator(acc_dims)
#
#     nvecs_loops += 'int c_idx = 0;'
#     nvecs_loops += 'auto curr_value_ptr = values;'
#     nvecs_loops += 'const Scalar *__restrict__ B_curr = col_indices[0] * n + B;'
#
#     if PREFETCH_B:
#         nvecs_loops += 'const Scalar *__restrict__ B_next = col_indices[1] * n + B;'
#         nvecs_loops += 'const Scalar *__restrict__ B_next_next = col_indices[2] * n + B;'
#         nvecs_loops += prefetch_B(acc_dims[1])
#         nvecs_loops += prefetch_B_next(acc_dims[1])
#         nvecs_loops += prefetch_B_next_next(acc_dims[1])
#
#
#     for id, pat in enumerate(m_supported_patterns):
#         nvecs_loops += gen_pattern_case(acc_dims, id, pat)
#
#     # switch_body += f'default:'
#     # default_case_body = Block(add_braces=True)
#     # default_case_body += 'std::cerr << "Unsupported pattern " << pattern << std::endl;'
#     # default_case_body += 'exit(-1);'
#     # switch_body += default_case_body
#
#     #nvecs_loops += switch_body
#     #nvecs_loops += pattern_loop
#
#     nvecs_loops += store_accumulator(acc_dims)
#
#     if PRECOMPUTE_B:
#         nvecs_loops += increment_B_ptrs(acc_dims[1])
#     else:
#         nvecs_loops += increment_B_ptr(acc_dims[1])
#
#     nvecs_loops += increment_C_ptr(acc_dims[1])
#     function_body += nvecs_loops
#
#     f.write(f'template<typename _Vec>\n')
#     f.write(f'struct PatExecutor<_Vec, {vec_height}> {{\n')
#     f.write(f'    using Vec = _Vec;\n')
#     f.write(f'    using Scalar = typename Vec::Scalar;\n')
#     f.write(f'    using VecType = typename Vec::Type;\n\n')
#     f.write(supported_pattern_getter)
#     f.write(supported_pattern_map)
#     f.write(supported_pattern_unmap)
#     f.write(pattern_popcount_map)
#     f.write(f'    static int sop_accumulator_width_in_vecs() {{ return {acc_dims[1]}; }};\n\n')
#     f.write(f'    static int number_of_patterns() {{ return {len(m_supported_patterns)}; }}\n\n')
#     f.write(f'    static int panel_height() {{ return {vec_height}; }}\n\n')
#     f.write(function_header + function_body.emit(2))
#     f.write(f'\n}};\n\n')
#
#
# import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
# with open(f'{SCRIPT_DIR}/../../cpp_testbed/demo/SpMM_methods/pat_executor.h', 'w+') as f:
#     f.write(FILE_HEADER)
    #gen_for_vec_height(f, [16, 1], SUPPORTED_PATTERNS_16, 16)
    # gen_for_vec_height(f, [8, 2], SUPPORTED_PATTERNS_8, 8)
    # gen_for_vec_height(f, [4, 4], SUPPORTED_PATTERNS_4, 4)

