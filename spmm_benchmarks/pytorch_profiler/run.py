import torch
from torch.profiler import profile, ProfilerActivity
from .parse import parse_profile_results


def torch_profiler(task_spec, profile_mapping):
    schedule = torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=2)

    with profile(
        with_stack=True,
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule,
    ) as p:
        for idx in range(4):
            setup_f = task_spec.get("setup", None)
            run_f = task_spec["stmt"]

            if setup_f:
                eval(setup_f, globals(), task_spec["globals"])

            p.start()
            eval(run_f, globals(), task_spec["globals"])
            p.step()
            p.stop()

    return {
        "results": parse_profile_results(p, profile_mapping),
        "sub_label": task_spec['sub_label'],
        "label": task_spec['label'],
        "description": task_spec['description']
    }
