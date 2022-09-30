import torch

from spmm_benchmarks.utils.cache import cached_return_by_input_id


class DDTTrace:
    def __init__(self, tuples, inner_iteration_ptr, cache_id=None):
        self.tuples = tuples
        self.inner_iteration_ptr = inner_iteration_ptr

        self.outer_iterations = len(inner_iteration_ptr) - 1
        self.inner_iterations = len(tuples)

        self._cache_id = cache_id

    def num_inner_iterations_for(self, outer):
        return self.inner_iteration_ptr[outer + 1] - self.inner_iteration_ptr[outer]

    def inner_iteration_bounds_for(self, outer):
        return self.inner_iteration_ptr[outer], self.inner_iteration_ptr[outer + 1]


@cached_return_by_input_id(refresh_cache=False)
def spmx_trace_gen(A: torch.Tensor):
    assert A.layout == torch.sparse_csr

    tuples, inner_iteration_ptr = torch.ops.ddt_inspector.gen_spmx_trace_csr(A)
    return DDTTrace(tuples, inner_iteration_ptr, cache_id=A._cache_id + "_trace")
