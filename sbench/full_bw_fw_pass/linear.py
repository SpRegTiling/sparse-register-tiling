import itertools
import torch
from torch.profiler import record_function

from xformers import _is_sparse_available
from xformers.components.attention.core import SparseCS, _sparse_bmm
from sbench.utils.test import construct_test_tensors


def zero_grad(a: torch.Tensor, b: torch.Tensor):
    for p in [a, b]:
        if isinstance(p, SparseCS):
            p = p._mat._SparseCSRTensor__values

        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()


def dense_linear_pre(input: torch.Tensor, weight: torch.Tensor):
    with record_function("forward"):
        out = torch.norm(weight @ input)

    with record_function("backward"):
        out.backward()

        input_grad = input.grad
        weight_grad = weight.grad

    return out, input_grad, weight_grad  # Return grads just to trigger the backward pass


def dense_linear_post(input: torch.Tensor, weight: torch.Tensor):
    with record_function("forward"):
        out = torch.norm(input @ weight)

    with record_function("backward"):
        out.backward()

        input_grad = input.grad
        weight_grad = weight.grad

    return out, input_grad, weight_grad  # Return grads just to trigger the backward pass


@torch.jit.script
def _mkldnn_mul(a, b):
    print(a.dtype, b.dtype)
    a_mkldnn = a if a.is_mkldnn else a.to_mkldnn(a.dtype)
    b_mkldnn = b if b.is_mkldnn else b.to_mkldnn(b.dtype)

    print(a_mkldnn.shape, b_mkldnn.shape)
    bias = torch.zeros([b_mkldnn.size(0)], dtype=b.dtype).to_mkldnn(b.dtype)

    c_mkldnn = torch._C._nn.mkldnn_linear(a_mkldnn, b_mkldnn, bias)
    c = c_mkldnn if a.is_mkldnn else c_mkldnn.to_dense()

    return c


def mkldnn_linear_pre(input: torch.Tensor, weight: torch.Tensor):
    raise NotImplemented()
    return None, None, None  # Return grads just to trigger the backward pass


def mkldnn_linear_post(input: torch.Tensor, weight: torch.Tensor):
    print(torch.backends.mkldnn.enabled)

    with torch.backends.mkldnn.flags(enabled=False):
        with record_function("forward"):
            input_shape = input.shape
            out = torch.norm(_mkldnn_mul(input.view(-1, input_shape[-1]), weight).view(*input_shape))

        with record_function("backward"):
            out.backward()

            input_grad = input.grad
            weight_grad = weight.grad

    return out, input_grad, weight_grad  # Return grads just to trigger the backward pass


def sputnik_linear_pre(input: torch.Tensor, weight: torch.Tensor):
    assert isinstance(weight, SparseCS)
    assert _is_sparse_available

    with record_function("forward"):
        out = torch.norm(weight.spmm(input))

    with record_function("backward"):
        out.backward()

        input_grad = input.grad
        weight_grad = weight._mat._SparseCSRTensor__values.grad
        torch.cuda.synchronize()

    return out, input_grad, weight_grad


def sputnik_linear_post(input: torch.Tensor, weight: torch.Tensor):
    assert isinstance(weight, SparseCS)
    assert _is_sparse_available

    with record_function("forward"):
        input_t = input.transpose(-1, -2).contiguous()
        out = torch.norm(weight.spmm(input_t)).transpose(-1, -2)

    with record_function("backward"):
        out.backward()

        input_grad = input.grad
        weight_grad = weight._mat._SparseCSRTensor__values.grad
        torch.cuda.synchronize()

    return out, input_grad, weight_grad


def run_sparse_benchmarks(runner,
                           f, setup,
                           input_shape, weight_shape,
                           dense_types: list,
                           sparse_types: list, sparsities: list,
                           label, description,
                           device: str,
                           requires_grad=True, precompute_transpose=False):
    is_setup_required = setup is not None

    if type(f) == str:
        f_name = lambda type: f.format(type)
    elif callable(f):
        f_name = lambda type: f.__name__
    else:
        raise NotImplemented("Unsupported function or pattern")

    results = []

    _globals = {}
    task_spec = dict(
        label=label,
        globals=_globals,
        description=description
    )

    if is_setup_required:
        setup_name = setup.__name__
        _globals[setup_name] = setup
        task_spec["setup"] = f'{setup_name}(input, weight)'

    _globals["input"] = construct_test_tensors("dense", input_shape,
                                               device=device, requires_grad=requires_grad)

    for dense_type in dense_types:
        task_spec["stmt"] = f'{f_name(dense_type)}(input, weight)'
        _globals[f_name(dense_type)] = globals()[f_name(dense_type)]

        _globals["weight"] = construct_test_tensors(dense_type, weight_shape,
                                                    device=device, requires_grad=requires_grad)

        task_spec["sub_label"] = f"dense {dense_type}"
        results.append(runner[dense_type](task_spec))

    for sparse_type, sparsity in itertools.product(sparse_types, sparsities):
        task_spec["stmt"] = f'{f_name(sparse_type)}(input, weight)'
        _globals[f_name(sparse_type)] = globals()[f_name(sparse_type)]

        _globals["weight"] = construct_test_tensors(sparse_type, weight_shape, sparsity,
                                                    device=device, requires_grad=requires_grad,
                                                    precompute_transpose=precompute_transpose)

        task_spec["sub_label"] = f"sparsity {sparse_type}: {sparsity:0.2f}"
        results.append(runner[sparse_type](task_spec))

    return results
