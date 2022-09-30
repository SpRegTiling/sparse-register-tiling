import torch
import matplotlib.pyplot as plt
from xformers.sparse.csr_tensor import SparseCSRTensor
from spmm_benchmarks.loaders.load import load_dense

mtx = '/sdb/datasets/hybrid/body_encoder_layer_5_self_attention_multihead_attention_q_fully_connected_sp_80.mtx'
mtx = load_dense(mtx)
plt.spy(mtx)
plt.show(block=True)
