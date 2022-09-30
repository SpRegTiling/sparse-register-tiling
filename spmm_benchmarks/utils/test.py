import torch
from xformers.components.attention.core import SparseCS, _create_random_sparsity
from xformers.sparse.csr_tensor import SparseCSRTensor
from torch.utils.mkldnn import MkldnnLinear


tensor_types = [
    "dense",
    "sputnik",
    "pytorch_sparse"
]


def construct_test_tensors(type, shape,
                           sparsity=None,
                           requires_grad=False,
                           precompute_transpose=False,
                           device=None):
    if sparsity is not None:
        density = (1-sparsity)

    if type == "dense":
        a = torch.rand(*shape, device=device, requires_grad=requires_grad)
        if sparsity is not None:
            a[a < density] = 0
    elif type == "sputnik":
        a = _create_random_sparsity(torch.rand(*shape, device=device), sparsity)
        a = SparseCS(a, device, precompute_transpose=precompute_transpose)
        torch.Tensor.requires_grad_(a._mat)
    elif type == "csr":
        a = _create_random_sparsity(torch.rand(*shape, device=device), sparsity)
        a = SparseCSRTensor.from_dense(a)
    elif type == "pytorch_sparse":
        a = _create_random_sparsity(torch.rand(*shape, device=device), sparsity)
        a = a.to_sparse()
        a.requires_grad = requires_grad
    elif type == "mkldnn":
        a = MkldnnLinear(torch.nn.Linear(*shape, bias=False, device=device))
    else:
        raise NotImplemented(f'{type} Tensor type not supported')

    return a
