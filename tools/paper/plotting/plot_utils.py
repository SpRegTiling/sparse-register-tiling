import altair_saver
import matplotlib.pyplot as plt
import os;

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

PLOT_DIR = "/sdb/paper_plots/"
RESULTS_DIR = "/sdb/paper_results/"


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


def load_dlmc_df(subdir, nthreads=None, bcols=None):
    assert nthreads or bcols

    if nthreads is not None and bcols is not None:
        return pd.read_csv(RESULTS_DIR + '/cache/' + subdir + f'/dlmc_bcols_{bcols}_nthreads_{nthreads}.csv')
    elif nthreads is not None:
        return pd.read_csv(RESULTS_DIR + '/cache/' + subdir + f'/dlmc_nthreads_{nthreads}.csv')
    elif bcols is not None:
        return pd.read_csv(RESULTS_DIR + '/cache/' + subdir + f'/dlmc_bcols_{bcols}.csv')


def create_chart_grid(charts, row_width):
    charts_merged = None
    charts_row = None

    col = 0
    for i in range(0, len(charts)):
        col = i % row_width
        if col == 0:
            charts_merged = charts_row if charts_merged is None else charts_merged & charts_row
            charts_row = None

        charts_row = charts[i] if charts_row is None else charts_row | charts[i]

    if col:
        charts_merged = charts_row if charts_merged is None else charts_merged & charts_row
    return charts_merged


def chart_save(chart, filename):
    filepath = PLOT_DIR + filename
    filepath = filepath.replace(".png", "") + ".png"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    altair_saver.save(chart, filepath, fmt="png", scale_fator=4)


def plot_save(filename):
    filepath = PLOT_DIR + filename
    filepath = filepath.replace(".pdf", "") + ".pdf"
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath)
