from operator import attrgetter
import itertools
from typing import List
from torch.autograd.profiler_util import FunctionEvent, DeviceType
from collections import defaultdict

already_printed = set()
def __dump(event, indent=1):
    if event in already_printed or event is None:
        return

    print("-" * indent, event.key, event.cuda_time_total, event.cpu_time_total)
    already_printed.add(event)

    for event in event.cpu_children:
        if event in already_printed: continue
        print("-"*indent, event.key, event.cuda_time_total, event.cpu_time_total)
        __dump(event, indent+1)


def _dump(results):
    global already_printed
    already_printed = set()
    for event in results:
        __dump(event)


def _parse_profile_results(results, mapping):
    mapped = defaultdict(lambda: 0)
    remaining_cuda_time = 0

    for event in results:
        if event.key in mapping.keys():
            event_mapping = mapping[event.key]

            if type(event_mapping) is dict:
                children_remaining, children_mapped = \
                    _parse_profile_results(event.cpu_children, event_mapping["children"])
                mapped.update(children_mapped)
                mapped[event_mapping["remaining"]] += children_remaining
            else:
                mapped[mapping[event.key]] += event.cuda_time_total
        else:
            matched = False
            for prefix_mapping in mapping.get("*", []):
                if prefix_mapping["prefix"] in event.key:
                    children_remaining, children_mapped = \
                        _parse_profile_results(event.cpu_children, prefix_mapping["children"])
                    mapped.update(children_mapped)
                    mapped[prefix_mapping["remaining"]] += children_remaining

            if not matched:
                remaining_cuda_time += event.cuda_time_total

    return remaining_cuda_time, mapped


def parse_profile_results(results, mapping):
    averages = results.key_averages()
    #_dump(averages)

    _, mapped = _parse_profile_results(averages, mapping)
    return mapped
