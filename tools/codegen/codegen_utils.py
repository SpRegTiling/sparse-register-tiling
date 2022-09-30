INDENTATION_DEFAULT = 2

"""
Internal Utils
"""


def _callable_array_addr(array):
    if type(array) is str:
        array_func = lambda x: f'&{array}[{x}]'
    elif callable(array):
        array_func = array
    else:
        raise Exception("Not a valid array type")
    return array_func


def _callable_array(array):
    if type(array) is str:
        array_func = lambda x: f'{array}[{x}]'
    elif callable(array):
        array_func = array
    else:
        raise Exception("Not a valid array type")
    return array_func


class Block:
    def __init__(self, add_braces=False):
        self.sub_elements = []
        self.add_braces = add_braces

    def emit(self, indentation):
        string = ' ' * indentation + '{\n' if self.add_braces else ""

        if self.add_braces:
            indentation += INDENTATION_DEFAULT

        for sub_element in self.sub_elements:
            if isinstance(sub_element, Block):
                extra_indent = 0 if isinstance(sub_element, ForLoop) else INDENTATION_DEFAULT
                string += sub_element.emit(indentation + extra_indent) + '\n'
            else:
                string += ' ' * indentation + sub_element + '\n'

        if self.add_braces:
            indentation -= INDENTATION_DEFAULT
            string += ' ' * indentation + '}'

        return string

    def __iadd__(self, other):
        if isinstance(other, Block):
            self.sub_elements.append(other)
        elif isinstance(other, list):
            self.sub_elements += other + ['']
        elif type(other) is str:
            self.sub_elements += [other]
        else:
            raise Exception(f'Cannot add {other} to block')
        return self

    def __str__(self):
        return str(self.sub_elements)


class ForLoop(Block):
    def __init__(self, init, cond, inc, unroll=None):
        super().__init__()
        self.inner_block = Block()
        if unroll is not None:
            self.sub_elements.append(f'#pragma GCC unroll {unroll}')
        self.sub_elements.append(f'for({init}; {cond}; {inc}) {{')
        self.sub_elements.append(self.inner_block)
        self.sub_elements.append('}')

    def __iadd__(self, other):
        self.inner_block += other
        return self


class Switch(Block):
    def __init__(self, value):
        super().__init__()
        self.inner_block = Block()
        self.sub_elements.append(f'switch({value}) {{')
        self.sub_elements.append(self.inner_block)
        self.sub_elements.append('}')

    def __iadd__(self, other):
        self.inner_block += other
        return self

"""
Codegen Objects
"""

class VecType:
    fields = ['x', 'y', 'z', 'w']
    scalar_mapping = {
        'float4': 'float'
    }

    def __init__(self, name, width):
        self.name = name
        self.width = width

    def emit_dot_function_definition(self):
        name = self.name
        scalar = self.scalar_mapping[name]
        func = f'__device__ __forceinline__ void dot(const {name} &x1, const {name} &x2, {scalar} &out) {{\n'
        for field in self.fields[:self.width]:
            func += f'  out += x1.{field} *  x2.{field};\n'
        func += "}\n"
        return func

    def emit_dot(self, x1, x2, out):
        return [f'dot({x1}, {x2}, {out});']

    def emit_fma_function_definition(self):
        name = self.name
        scalar = self.scalar_mapping[name]
        func = f'__device__ __forceinline__ void fma(const {scalar} &x1, const {name} &x2, {name} &out) {{\n'
        for field in self.fields[:self.width]:
            func += f'  out.{field} += x1 *  x2.{field};\n'
        func += "}\n"
        func += f'__device__ __forceinline__ void fma(const {name} &x1, const {scalar} &x2, {name} &out) {{\n'
        for field in self.fields[:self.width]:
            func += f'  out.{field} += x1.{field} *  x2;\n'
        func += "}\n"
        return func

    def emit_fma(self, x1, x2, out):
        return [f'fma({x1}, {x2}, {out});']


class RegTile:
    def __init__(self, name, reg_prefix, type_name, num_reg):
        self.name = name
        self.reg_prefix = reg_prefix
        self.num_reg = num_reg
        self.type_name = type_name
        assert 'float' in type_name

    def emit_initialization(self, default_value=None):
        lines = []

        if default_value is None:
            for i in range(0, self.num_reg, 16):
                reg_list = ", ".join([f'{self.reg_prefix}{j}' for j in range(i, min(self.num_reg, i+16))])
                lines.append(f'{self.type_name} {reg_list};')
        else:
            for i in range(self.num_reg):
                lines.append(f'{self.type_name} {self.reg_prefix}{i} = {default_value};')

        return lines

    def emit_shared_store_all(self, array, start_offset=0):
        array_func = _callable_array(array)
        write_locs = [array_func(f'{start_offset + reg_id}') for reg_id in range(self.num_reg)]
        return [f'{write_locs[reg_id]} = {self.reg_prefix}{reg_id};' for reg_id in range(self.num_reg)]

    def emit_dot(self, acc, other_reg_tile, reg_list=None):
        if reg_list is None:
            reg_list = range(self.num_reg)

        assert len(reg_list) == other_reg_tile.num_reg

        lines = []
        for other_reg_id, reg_id in enumerate(reg_list):
            lines += self.type.emit_dot(f'{self.reg_prefix}{reg_id}', f'{other_reg_tile.reg_prefix}{other_reg_id}', acc)
        return lines

    def reg(self, reg_id):
        return f'{self.reg_prefix}{reg_id}'


class VecRegTile(RegTile):
    def __init__(self, name, reg_prefix, vec_type, num_reg):
        super().__init__(name, reg_prefix, vec_type.name, num_reg)
        self.type = vec_type

    def emit_const_global_vec_load(self, reg_id, addr):
        assert reg_id < self.num_reg
        return [f'{self.reg_prefix}{reg_id} = global_const_load_float<{self.type_name}>({addr});']

    def emit_const_shared_vec_load(self, reg_id, addr):
        assert reg_id < self.num_reg
        return [f'{self.reg_prefix}{reg_id} = shared_const_load_float<{self.type_name}>({addr});']

    def emit_const_global_vec_load_all(self, array, start_offset):
        array_func = _callable_array_addr(array)

        lines = []
        for reg_id in range(self.num_reg):
            lines += self.emit_const_global_vec_load(
                reg_id, array_func(f'{start_offset} + {reg_id * self.type.width}'))

        return lines

    def emit_const_shared_vec_load_all(self, array, start_offset = 0):
        array_func = _callable_array_addr(array)

        lines = []
        for reg_id in range(self.num_reg):
            lines += self.emit_const_shared_vec_load(
                reg_id, array_func(f'{start_offset} + {reg_id * self.type.width}'))

        return lines
