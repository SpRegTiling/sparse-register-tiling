import altair as alt
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tools.paper.plotting.plot_utils import *

def filter(df, **kwargs):
    bool_index = None
    for key, value in kwargs.items():
        if isinstance(value, list):
            _bool_index = df[key].isin(value)
        else:
            _bool_index = df[key] == value
        if bool_index is None:
            bool_index = _bool_index
        else:
            bool_index = bool_index & _bool_index
    return df[bool_index]


BENCHMARK_NAME_MAP = {
    "FP32MobileNetV1": ("XNN Dense", 0.0),
    "FP32Sparse70MobileNetV1": ("XNN Sparse", 0.7),
    "FP32Sparse80MobileNetV1": ("XNN Sparse", 0.8),
    "FP32Sparse90MobileNetV1": ("XNN Sparse", 0.9),
    "FP32Sparse70MobileNetV1Nano": ("Sparse Reg Tiling", 0.7),
    "FP32Sparse80MobileNetV1Nano": ("Sparse Reg Tiling", 0.8),
    "FP32Sparse90MobileNetV1Nano": ("Sparse Reg Tiling", 0.9),
}


results = json.load(open('/sdb/paper_results/end2end/end2end_bench_v1.json'))
optimized_layers = list(range(3, 28, 2))

results_per_threadcount = defaultdict(lambda: {})
dense_baseline_times_per_threadcount = {}
sparse_baseline_times_per_threadcount = defaultdict(lambda: {})
total_time_per_threadcount = defaultdict(lambda: {})


def Label(x):
    if x == 0:
        return "XNN Dense"
    return f'{x[0]} ({x[1]})'


for benchmark in results["benchmarks"]:
    benchmark_name = BENCHMARK_NAME_MAP[benchmark["name"].split("/")[0]]
    threads = int(benchmark["name"].split("/")[1].split(":")[1])

    layer_times = []
    for key, value in benchmark.items():
        if "layer" in key:
            layer = int(key.split("_")[1])
            layer_times.append((layer, value))

    layer_times = sorted(layer_times, key=lambda x: x[0])
    layer_times = np.array([x[1] for x in layer_times])

    results_per_threadcount[threads][benchmark_name] = layer_times
    total_time_per_threadcount[threads][benchmark_name] = benchmark["real_time"]

    if benchmark_name[1] == 0.0:
        dense_baseline_times_per_threadcount[threads] = benchmark["real_time"]

    if benchmark_name[0] == "XNN Sparse":
        sparse_baseline_times_per_threadcount[threads][benchmark_name[1]] = benchmark["real_time"]

print(dense_baseline_times_per_threadcount)


for threads in sorted(results_per_threadcount.keys()):
    print("Dense", dense_baseline_times_per_threadcount[threads] / 1000)
    print(f"\nVs Dense, Threads: {threads}\n")
    for benchmark_name, time in total_time_per_threadcount[threads].items():
        print(Label(benchmark_name), time/1000, dense_baseline_times_per_threadcount[threads], dense_baseline_times_per_threadcount[threads] / time)


for threads in sorted(results_per_threadcount.keys()):
    print(f"\nVs Sparse, Threads: {threads}\n")
    for benchmark_name, time in total_time_per_threadcount[threads].items():
        if benchmark_name[1] == 0.0: continue
        print(Label(benchmark_name), time/1000, sparse_baseline_times_per_threadcount[threads][benchmark_name[1]], sparse_baseline_times_per_threadcount[threads][benchmark_name[1]] / time)


for threads in sorted(results_per_threadcount.keys()):
    fig, axs = plt.subplots(2, 1)

    LINE_STYLES = {
        "XNN Dense": '-',
        "XNN Sparse": ':',
        "Sparse Reg Tiling": '--',
    }

    COLORS = {
        0.0: 'black',
        0.7: 'red',
        0.8: 'blue',
        0.9: 'green',
    }

    for benchmark_name, times in results_per_threadcount[threads].items():
        print(benchmark_name, times)
        line_style = LINE_STYLES[benchmark_name[0]]
        color = COLORS[benchmark_name[1]]
        axs[0].plot(range(1, len(times)+1), times/1000, marker='x', label=Label(benchmark_name),
                    linestyle=line_style, color=color)

    for i in optimized_layers:
        # only one line may be specified; full height
        axs[0].axvline(x=i, color='b', label=None, linewidth=0.5)

    for benchmark_name, times in results_per_threadcount[threads].items():
        line_style = LINE_STYLES[benchmark_name[0]]
        color = COLORS[benchmark_name[1]]
        axs[1].plot(range(1, len(times)+1), np.cumsum(times)/1000, marker='x', label=Label(benchmark_name),
                    linestyle=line_style, color=color)

    for i in optimized_layers:
        # only one line may be specified; full height
        axs[1].axvline(x=i, color='b', label=None, linewidth=0.5)

    # for i in optimized_layers:
    #     # only one line may be specified; full height
    #     axs[2].axvline(x=i, color='b', label=None, linewidth=0.5)

    #baseline_times = results_per_threadcount[threads][("XNN Dense", 0.0)]

    # for benchmark_name, times in results_per_threadcount[threads].items():
    #     line_style = LINE_STYLES[benchmark_name[0]]
    #     color = COLORS[benchmark_name[1]]
    #     axs[2].plot(range(1, len(times)+1), baseline_times / times,
    #                 marker='x', label=Label(benchmark_name),
    #                 linestyle=line_style, color=color)
    #
    # axs[2].axhline(y=1, color='black', linestyle='-', linewidth=0.5)
    # axs[2].set_ylim(0, 3)

    axs[0].legend(loc='upper left')
    axs[1].legend(loc='upper left')
    #axs[2].legend(loc='upper left')

    axs[0].set_ylabel('Layer Time (ms)')
    axs[1].set_ylabel('Cumulative Time (ms)')
    axs[1].set_xlabel('Layer')

    plt.tight_layout()

    plot_save(f'end2end_bench_v1_{threads}.pdf')

