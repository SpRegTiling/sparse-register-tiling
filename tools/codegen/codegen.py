def generate_1d_tiling(unrolling_factor, intrinsic_type=512, file_dir='code.txt'):
    NEW_LINE = '\t\t\t\ \n'
    TAB = "\t"
    code = f"#define KERNEL_{intrinsic_type}_{unrolling_factor} for (int j = 0; j < n; j += INTRINSIC_SIZE * {unrolling_factor}) " + "{"
    code += NEW_LINE
    for i in range(unrolling_factor):
        code += TAB + f"__m{intrinsic_type} accumulator_vector{i} = _mm{intrinsic_type}_setzero_ps();" + NEW_LINE
    code += NEW_LINE

    code += TAB + "for(int l = row_offsets[i]; l < row_offsets[i + 1]; l++) {" + NEW_LINE
    TAB = '\t\t'
    code += TAB + f"__m{intrinsic_type} lhs_duplicated_val = " + NEW_LINE
    if intrinsic_type == 512:
        code += TAB + f"\t_mm{intrinsic_type}_broadcast_f32x2(_mm_broadcast_ss(&values[b * nonzeros + l]));" + NEW_LINE
    else:
        code += TAB + f"\t_mm{intrinsic_type}_broadcast_ss(&values[b * nonzeros + l]);"

    code += NEW_LINE
    code += TAB + "int column_index = column_indices[l];" + NEW_LINE
    code += NEW_LINE

    for i in range(unrolling_factor):
        code += TAB + f"__m{intrinsic_type} rhs_row_vals{i} = _mm{intrinsic_type}_loadu_ps( " + NEW_LINE
        code += TAB + f"\t&dense_matrix[b * k * n + column_index * n + j + INTRINSIC_SIZE * {i}]); " + NEW_LINE

    code += NEW_LINE

    for i in range(unrolling_factor):
        code += TAB + f"accumulator_vector{i} = _mm{intrinsic_type}_fmadd_ps( " + NEW_LINE
        code += TAB + f"\tlhs_duplicated_val, rhs_row_vals{i}, accumulator_vector{i});" + NEW_LINE

    TAB = "\t"
    code += TAB + "}" + NEW_LINE

    for i in range(unrolling_factor):
        code += TAB + f"_mm{intrinsic_type}_storeu_ps( " + NEW_LINE
        code += TAB + f"&output_matrix[b * m * n + i * n + j + INTRINSIC_SIZE * {i}], accumulator_vector{i});" + NEW_LINE
    code += "}\n"

    with open(file_dir, 'w') as file:
        file.write(code)


for intrinsic in [256, 512]:
    for unrolling_factor in [2, 4, 8, 16, 32, 64]:
        generate_1d_tiling(unrolling_factor, intrinsic, f"kernel_{intrinsic}_{unrolling_factor}.cpp")
