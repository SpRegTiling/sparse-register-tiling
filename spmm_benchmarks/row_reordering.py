import os
import torch
import zlib
import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


from xformers.sparse.csr_tensor import SparseCSRTensor

rows, cols, row_ptrs, col_indices = \
    torch.ops.spmm_benchmarks.load_smtx(
        '/sdb/codegen/dlmc/rn50/magnitude_pruning/0.8/bottleneck_1_block_group1_1_1.smtx')

weight = {}
golden = {}
nnz = len(col_indices)


def similarity(row1, row2):
    return (row1 * row2).sum()


values = torch.ones(nnz)

# Unsqueezing since the spmm is batched
weight_csr = SparseCSRTensor(
    row_offsets=row_ptrs,
    column_indices=col_indices,
    values=values.unsqueeze(0),
    shape=(1, rows, cols)
)

weight_dense = SparseCSRTensor._to_dense(weight_csr).squeeze(0)
clusters = range(rows)

best_pair = -1
best_pair_similarity = 0

row_ordering = []
visited = set()

for row in weight_dense:
    for alt_row in weight_dense:
        sim = similarity(row, alt_row)
        if sim > best_pair:
            best_pair = (row, alt_row)
            best_pair_similarity = sim

row1, row2 = best_pair
row1 
