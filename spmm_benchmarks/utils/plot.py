import numpy as np
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import os

from collections import defaultdict, OrderedDict
from torch.utils.benchmark.utils.compare import Compare
from altair_saver import save

colors = [
   '#003f5c',
   '#2f4b7c',
   '#665191',
   '#a05195',
   '#d45087',
   '#f95d6a',
   '#ff7c43',
   '#ffa600',
]


class TorchProfiler:
    @staticmethod
    def partition_sparse(_group):
        _sparse = defaultdict(lambda: [])

        for row in _group:
            sub_label = row["sub_label"]
            if 'sparsity' in sub_label:
                label, sparsity = sub_label.split(':')
                label = label.split(" ")[-1].strip()
                _sparse[label].append(row["results"])
                _sparse[label][-1]["sparsity"] = sparsity

        return _sparse

    @staticmethod
    def partition_dense(_group):
        _sparse = defaultdict(lambda: [])

        for row in _group:
            sub_label = row["sub_label"]
            if 'dense' in sub_label:
                results = row["results"]
                results["sparsity"] = 1
                return results

        return _sparse

    @staticmethod
    def partition_shapes(_group):
        shapes = defaultdict(lambda: [])
        for row in _group:
            shapes[row["description"]].append(row)
        return shapes

    @staticmethod
    def group_by_label(results):
        _groups = defaultdict(lambda: [])

        for row in results:
            _groups[row["label"]].append(row)

        return _groups


def sanitize_path_string(string):
    return string.replace(" ", "_").replace("/", "_").replace("-", "_").replace(":", "").replace(",", "")


def stack_plot(results, path: str = 'plots', title=None):
    path = path + "/" + ("" if title is None else sanitize_path_string(title))
    if not os.path.exists(path): os.makedirs(path)

    groups = TorchProfiler.group_by_label(results)
    shape_results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [])))

    for group_name in groups:
        group = groups[group_name]

        for shape_name, shape_data in TorchProfiler.partition_shapes(group).items():
            sparse = TorchProfiler.partition_sparse(shape_data)
            assert len(sparse) == 1

            for method, data in sparse.items():
                shape_results[shape_name][group_name] = data

    for shape_name, results in shape_results.items():
        df_labels = []
        dfs = []

        for group_name, group_results in results.items():
            df = pd.DataFrame(group_results).set_index('sparsity')
            dfs.append(df)
            df_labels.append(group_name.replace("bmm_", "").replace("pass_", ""))

        def prep_df(df, name):
            df = df.stack().reset_index()
            df.columns = ['sparsity', 'kernel', 'time']
            df['label'] = name
            return df

        dfs = [prep_df(df, label) for df, label in zip(dfs, df_labels)]
        df = pd.concat(dfs)

        chart = alt.Chart(df).mark_bar().encode(
            # tell Altair which field to group columns on
            x=alt.X('label:N', title=None),
            # tell Altair which field to use as Y values and how to calculate
            y=alt.Y('sum(time):Q', axis=alt.Axis(grid=False, title='Running time (us)')),
            # tell Altair which field to use to use as the set of columns to be  represented in each group
            column=alt.Column('sparsity:N', title='Sparsity'),
            # tell Altair which field to use for color segmentation
            color=alt.Color('kernel:N',
                scale=alt.Scale(
                    # make it look pretty with an enjoyable color pallet
                    range=['#003f5c', '#444e86', '#955196', '#dd5182', '#ff6e54', '#ffa600'],
                ))
        ).configure_view(
            # remove grid lines around column clusters
            strokeOpacity=0
        ).properties(title=title)

        # Sanitize the filename
        filename = shape_name + "_stacked"
        filename = sanitize_path_string(filename)

        filepath = path + "/" + filename + '.png'
        print('Saving:', filepath)
        save(chart, filepath, scale_factor=2.0)


def stack_area_plot_torch_profiler(results, path: str = 'plots', title=None):
    path = path + "/" + ("" if title is None else sanitize_path_string(title))
    if not os.path.exists(path): os.makedirs(path)

    groups = TorchProfiler.group_by_label(results)
    shape_results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: []))))

    for group_name in groups:
        group = groups[group_name]

        for shape_name, shape_data in TorchProfiler.partition_shapes(group).items():
            sparse = TorchProfiler.partition_sparse(shape_data)
            assert len(sparse) == 1

            for method, data in sparse.items():
                shape_results[shape_name]["sparse"][group_name] = data

        for shape_name, shape_data in TorchProfiler.partition_shapes(group).items():
            dense = TorchProfiler.partition_dense(shape_data)
            shape_results[shape_name]["dense"][group_name] = dense

    SCALE = 1
    for shape_name, sd_results in shape_results.items():
        def prep_df(df, name):
            df = df.stack().reset_index()
            df.columns = ['sparsity', 'kernel', 'time']
            df['label'] = name
            return df

        df_labels = []
        dfs = []

        results = sd_results["sparse"]
        for group_name, group_results in results.items():
            df = pd.DataFrame(group_results).set_index('sparsity')
            dfs.append(df)
            df_labels.append(group_name.replace("bmm_", "").replace("pass_", ""))

        dfs = [prep_df(df, label) for df, label in zip(dfs, df_labels)]
        df = pd.concat(dfs)

        assert len(groups) == 1
        sparse_chart = alt.Chart(df, width=240*2*SCALE, height=180*2*SCALE).mark_area().encode(
            # tell Altair which field to group columns on
            x=alt.X('sparsity:N', title='Sparsity'),
            # tell Altair which field to use as Y values and how to calculate
            y=alt.Y('sum(time):Q', axis=alt.Axis(grid=False, title='Running time (us)')),
            # tell Altair which field to use for color segmentation
            color=alt.Color('kernel:N', scale=alt.Scale(range=colors))
        )

        dfs = []
        df_labels = []

        results = sd_results["dense"]
        for group_name, group_results in results.items():
            df = pd.DataFrame([group_results]).set_index('sparsity')
            dfs.append(df)
            df_labels.append(group_name.replace("bmm_", "").replace("pass_", ""))

        dfs = [prep_df(df, label) for df, label in zip(dfs, df_labels)]
        df = pd.concat(dfs)

        dense_chart = alt.Chart(df, width=60*SCALE, height=180*2*SCALE).mark_bar().encode(
            # tell Altair which field to group columns on
            x=alt.X('sparsity:N', title='Dense'),
            # tell Altair which field to use as Y values and how to calculate
            y=alt.Y('sum(time):Q', axis=None),
            # tell Altair which field to use for color segmentation
            color=alt.Color('kernel:N', scale=alt.Scale(range=colors))
        )

        chart = alt.hconcat(sparse_chart, dense_chart)\
            .resolve_scale(y='shared') \
            .properties(title=f'{title} ({shape_name})')

        # Sanitize the filename
        filename = shape_name + "_area_stacked.png"
        filename = sanitize_path_string(filename)

        filepath = path + "/" + filename
        print('Saving:', filepath)
        save(chart, filepath, vega_cli_options=["-s 6"])


def speedup_plot_torch_benchmark(results: Compare, path: str = 'plots', title=None):
    path = path + "/" + ("" if title is None else sanitize_path_string(title))
    if not os.path.exists(path): os.makedirs(path)

    groups = results._group_by_label(results._results)

    def _partition_shapes(_group):
        shapes = defaultdict(lambda: [])
        for row in group:
            shapes[row.description].append(row)
        return shapes

    def _partition_sparse_dense(_group):
        dense = None
        sparse = defaultdict(lambda: {})

        for row in group:
            sub_label = row.task_spec.sub_label
            if 'dense' in sub_label:
                dense = row.mean
            if 'sparsity' in sub_label:
                label, sparsity = sub_label.split(':')
                label = label.split(" ")[-1].strip()
                sparse[label][float(sparsity)] = row.mean

        sparse_iterator = iter(sparse)
        sparsities = set(sparse[next(sparse_iterator)].keys())
        while (x := next(sparse_iterator, None)) is not None:
            assert sparsities == set(sparse[x].keys())

        return dense, sparse

    shape_results = defaultdict(lambda: defaultdict(lambda: {}))
    for name, group in groups.items():
        for shape_name, shape_data in _partition_shapes(group).items():
            results = shape_results[shape_name]
            dense, sparse = _partition_sparse_dense(shape_data)

            for method, data in sparse.items():
                od = OrderedDict(sorted(data.items()))
                sparsities, times = zip(*od.items())

                results["Sparsity"] = sparsities
                results[f'{method} {name}'] = dense / np.array(times)

    for shape_name, results in shape_results.items():
        plt.gcf().set_figwidth(6)
        plt.gcf().set_figheight(6)
        plt.title(f'{shape_name}')

        data = pd.DataFrame(results)
        sns.lineplot(x='Sparsity', y='value', hue='variable',
                     data=pd.melt(data, ['Sparsity']))

        plt.ylabel('Speed-up (over dense)')

        # Sanitize the filename
        filename = shape_name + "_speed_up.png"
        filename = sanitize_path_string(filename)

        plt.gcf().savefig(path + "/" + filename)
        plt.clf()
