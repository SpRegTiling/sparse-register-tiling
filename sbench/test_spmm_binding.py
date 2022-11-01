import torch
import xformers

import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
from sbench.loaders.load import load_dense, load_csr, load_coo

matrix = load_csr('/sdb/codegen/dlmc/transformer/magnitude_pruning/0.7/body_decoder_layer_0_encdec_attention_multihead_attention_q_fully_connected.smtx')
B = torch.rand(matrix.shape[1], 128)

# Sparse call
print(torch.ops.sparse.spmm_csr_c_1d_256(matrix.values(), matrix.crow_indices(), matrix.col_indices(), B))

print(torch.ops.sparse.spmm_csr_c_1d_512(matrix.values(), matrix.crow_indices(), matrix.col_indices(), B))

# Visually compare with dense
print(matrix.to_dense() @ B)
