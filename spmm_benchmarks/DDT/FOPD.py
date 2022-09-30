import trace
import torch

from .trace import DDTTrace
from spmm_benchmarks.utils.cache import cached_return_by_input_id


class ConsecutiveFOPD:
    def __init__(self, fopd_tuples, fopd_inner_iter_ptrs, dim, cache_id=None):
        self.tuples = fopd_tuples
        self.inner_iteration_ptr = fopd_inner_iter_ptrs
        self.dim = dim
        self._cache_id = cache_id


@cached_return_by_input_id(refresh_cache=False, kwargs_to_check=["dim"])
def compute_consecutive_fopds(trace: DDTTrace, **kwargs) -> ConsecutiveFOPD:
    dim = kwargs["dim"]
    fopd_tuples, fopd_inner_iter_ptrs = torch.ops.ddt_inspector.compute_consecutive_fopds(
        trace.tuples, trace.inner_iteration_ptr, dim)

    return ConsecutiveFOPD(fopd_tuples, fopd_inner_iter_ptrs, dim, cache_id=trace._cache_id + f'_{dim}')


@cached_return_by_input_id(refresh_cache=False, kwargs_to_check=["access_function", "max_consecutive"])
def hist_mine_consecutive_fopds(trace: ConsecutiveFOPD, **kwargs) -> ConsecutiveFOPD:
    access_function = kwargs["access_function"]
    max_consecutive = kwargs["max_consecutive"]
    hist = torch.ops.ddt_inspector.hist_mine_consecutive_fopds(trace.tuples, trace.inner_iteration_ptr,
                                                               access_function, max_consecutive)

    return hist
