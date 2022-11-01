import torch
from torch.profiler import profile, ProfilerActivity
from .parse import parse_profile_results


def profile_func(func, wait=1, warmup=1, active=2):
    def wrap(*args, **kwargs):
        schedule = torch.profiler.schedule(wait=wait, warmup=warmup, active=active)

        with profile(
            with_stack=True,
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule,
        ) as p:
            for i in range(wait + warmup + active):
                out = func(*args, **kwargs)
                p.step()

        return out, p

    return wrap
