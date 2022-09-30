import numpy  as np
from collections import namedtuple
from scipy.sparse import csr_matrix
from scipy.io import mmwrite
import argparse

CSRPattern = namedtuple('CSRPattern',
                        ['nrows', 'ncols', 'nnz', 'row_ptrs', 'col_indices'])


# Read the custom sputnik data file structure, its a pattern only file
# format so contains no values
#   line 1: nrows, ncols, nnz
#   line 2: row_ptrs ... (space delimited)
#   line 3: col_indices ... (space delimited)
def read_pattern(filepath):
    with open(filepath) as file:
        lines = [file.readline() for _ in range(3)]
        nrows, ncols, nnz = [int(x) for x in lines[0].split(',')]
        return CSRPattern(nrows=nrows, ncols=ncols, nnz=nnz,
                          row_ptrs=np.fromstring(lines[1], dtype=int, sep=" "),
                          col_indices=np.fromstring(lines[2], dtype=int, sep=" ")
                          )
    return None


# Convert the pattern to a scipy CSR matrix with 1s for all the values to
# enable using scipy libraries/plotting/etc.
def pattern_to_scipy_csr(csr_pattern: CSRPattern):
    nnz = len(csr_pattern.col_indices)
    return csr_matrix(([1] * nnz, csr_pattern.col_indices, csr_pattern.row_ptrs),
                      (csr_pattern.nrows, csr_pattern.ncols))


parser = argparse.ArgumentParser(description='Convert SMTX to MTX')
parser.add_argument('smtx_file')
parser.add_argument('mtx_file')
args = parser.parse_args()

mmwrite(args.mtx_file, pattern_to_scipy_csr(read_pattern(args.smtx_file)))
