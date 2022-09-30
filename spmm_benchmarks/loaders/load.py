import xformers     # pull in  torch.ops.spmm_benchmarks.load_smtx
import torch
import os
import scipy
import numpy as np
from scipy.io import mmread
from scipy import sparse


def _cache_inplace(fn):
    def wrap(path, target_type):
        _, extension = os.path.splitext(path)
        cache_file = path.replace(extension, f'_{target_type}.pt')
        if os.path.exists(cache_file):
            try:
                return torch.load(cache_file)
            except Exception as e:
                print(e)

        mat = fn(path, target_type)
        torch.save(mat, cache_file)
        return mat
    return wrap


@_cache_inplace
def _load_smtx(path, target_type):
    print(path)
    rows, cols, row_ptrs, col_indices = torch.ops.spmm_benchmarks.load_smtx(path)
    csr = torch.sparse_csr_tensor(
        row_ptrs, col_indices, torch.ones(len(col_indices)),
        size=(rows, cols))

    if target_type == "dense":
        return csr.to_dense()
    elif target_type == "csr":
        return csr
    elif target_type == "coo":
        return csr.to_sparse_coo().coalesce()
    else:
        raise NotImplemented(f'loading {target_type} type tensor from smtx')


@_cache_inplace
def _load_mtx(path, target_type):
    print(path)
    mtx = mmread(path)

    if type(mtx) == np.ndarray:
        tensor = torch.tensor(mtx)
    else:
        indices = torch.stack((torch.Tensor(mtx.row), torch.Tensor(mtx.col)))
        tensor = torch.sparse_coo_tensor(indices, torch.Tensor(mtx.data), size=mtx.shape)

    if target_type == "dense":
        return tensor.to_dense() if tensor.layout != torch.strided else tensor
    elif target_type == "csr":
        return tensor.to_sparse_csr()
    elif target_type == "coo":
        return tensor.to_sparse_coo().coalesce()
    else:
        raise NotImplemented(f'loading {target_type} type tensor from mtx')


def _load(path, target_type):
    filepath, extension = os.path.splitext(path)
    loader = {
        '.smtx': _load_smtx,
        '.mtx': _load_mtx
    }.get(extension, None)
    if loader is None: raise NotImplementedError(f'Unsupported file type {extension}')
    mtx = loader(path, target_type)

    filename = filepath.split('/')[-1]
    setattr(mtx, "_cache_id", f'{filename}_{extension[1:]}_{target_type}')

    return mtx


def load_coo(path) -> torch.Tensor:
    return _load(path, "coo")


def load_csr(path) -> torch.Tensor:
    return _load(path, "csr")


def load_dense(path) -> torch.Tensor:
    return _load(path, "dense")
