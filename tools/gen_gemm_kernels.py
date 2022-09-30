

FUNCTION_HEADER = """
template<typename VecType, typename StorageTypes>
__inline __attribute__((__always_inline__)) void _coo_block_{m}_{k}_x_{n}(
        int b_vecs,
        int m, int k, int n, int _i, int j,
        const typename StorageTypes::Scalar *__restrict__ values,
        const typename StorageTypes::Index *__restrict__ row_indices,
        const typename StorageTypes::Index *__restrict__ column_indices,
        const typename StorageTypes::Scalar *__restrict__ B,
        typename StorageTypes::Scalar *__restrict__ C)
{{
    typename StorageTypes::Scalar * C_ptrs[{m}];
    const typename StorageTypes::Scalar * B_ptrs[{k}];
"""


def def_c(dims):
    lines = []
    for k in range(dims[2]):
        lines.append(f'    VecType cVecK{k};')
    return "\n".join(lines) + "\n\n"


def load_a_row_indices(dims):
    lines = [
        f'    #pragma ivdep',
        f'    #pragma vector nontemporal (row_indices)',
        f'    #pragma prefetch row_indices:_MM_HINT_T1',
        f'    #pragma temporal (C_ptrs)',
        f'    for (int i = 0; i < {dims[0]}; i ++) {{',
        f'        C_ptrs[i] = (typename StorageTypes::Scalar *) uintptr_t(C) + uintptr_t(row_indices[i]) * n;',
        f'    }}',
    ]
    return "\n".join(lines) + "\n\n"


def load_b_row_indices(dims):
    lines = [
        f'    #pragma ivdep',
        f'    #pragma vector nontemporal (column_indices)',
        f'    #pragma prefetch column_indices:_MM_HINT_T1',
        f'    #pragma temporal (B_ptrs)',
        f'    for (int i = 0; i < {dims[1]}; i ++) {{',
        f'        B_ptrs[i] = (const typename StorageTypes::Scalar *) uintptr_t(B) + uintptr_t(column_indices[i]) * n;',
        f'    }}',
    ]
    return "\n".join(lines) + "\n\n"


def def_ab_row_indices(dims):
    lines = []
    for i in range(dims[1]):
        line = f'    VecType aVec{i};'
        lines.append(line)

    return "\n".join(lines) + "\n\n"


def def_load_b(dims):
    lines = []
    for i in range(dims[1]):
        for k in range(dims[2]):
            lines.append(f'       bVec{i}{k}.load(&B[bRow{i} + {k} * VecType::size()]);')

    return "\n".join(lines) + "\n\n"


def def_process_c_row(row_i, dims):
    lines = []

    return "\n".join(lines) + "\n\n"


def def_process_c(dims):
    lines = []
    lines.append(f'    #pragma unroll')
    lines.append(f'    for (int b_vec = 0; b_vec < b_vecs; b_vec += {dims[2]}) {{')

    for k in range(dims[2]):
        lines.append(f'       VecType bVec{k};')

    for i in range(dims[0]):
        for k in range(dims[2]):
            lines.append(f'       VecType cVec{i}{k};')

    for i in range(dims[0]):
        for k in range(dims[2]):
            lines.append(f'       cVec{i}{k}.load(C_ptrs[{i}] + {k} * VecType::size());')

    for col_i in range(dims[1]):
        for k in range(dims[2]):
            lines.append(f'       bVec{k}.load(B_ptrs[{col_i}] + {k} * VecType::size());')

        for i in range(dims[0]):
            lines.append(f'       aVec{i} = VecType(values[{col_i} * {dims[0]} + {i}]);')
            for k in range(dims[2]):
                lines.append(f'       cVec{i}{k} = mul_add(aVec{i}, bVec{k}, cVec{i}{k});')

    for i in range(dims[0]):
        for k in range(dims[2]):
            lines.append(f'       cVec{i}{k}.store(C_ptrs[{i}] + {k} * VecType::size());')

    for i in range(dims[0]):
        lines.append(f'       C_ptrs[{i}] += VecType::size() * {dims[2]};')

    for i in range(dims[1]):
        lines.append(f'       B_ptrs[{i}] += VecType::size() * {dims[2]};')

    lines.append(f'    }}')
    return "\n".join(lines) + "\n\n"


DIMS = [
    (4,  4, 1),
    (4,  4, 2),
    (4,  4, 4),
    (4,  8, 1),
    (4,  8, 2),
    (4,  8, 4),
    (4, 10, 1),
    (4, 10, 2),
    (4, 10, 4),
    (4, 12, 1),
    (4, 12, 2),
    (4, 12, 4),
    (4, 16, 1),
    (4, 16, 2),
    (4, 16, 4),
    (4, 32, 1),
    (4, 32, 2),
    (4, 32, 4),
    (4, 64, 1),
    (4, 64, 2),
    (6,  6, 1),
    (6,  6, 2),
    (6,  8, 1),
    (6,  8, 2),
    (6, 10, 1),
    (6, 10, 2),
    (6, 10, 4),
    (6, 16, 1),
    (6, 16, 2),
    (8,  8, 1),
    (8,  8, 2),
    (8, 16, 1),
    (8, 16, 2),
]

import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

with open(f'{SCRIPT_DIR}/../cpp_testbed/demo/dense_kernels.hpp', 'w+') as f:

    for dims in DIMS:
        function = \
            FUNCTION_HEADER.format(m=dims[0], k=dims[1], n=dims[2]) + \
            def_c(dims) + \
            load_a_row_indices(dims) + \
            load_b_row_indices(dims) + \
            def_ab_row_indices(dims) + \
            def_process_c(dims)

        function = function.rstrip() + "\n}\n\n"
        f.write(function)
