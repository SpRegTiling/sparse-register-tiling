import os; SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
import sys; sys.path.insert(0,f'{SCRIPT_DIR}/../')
import pandas as pd
import seaborn as sns
import altair as alt
from plot_utils import *


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


df = pd.read_csv(SCRIPT_DIR + '/.cache/dlmc_part1_df_merged_postprocessed_v2.csv')
print(df["name"].unique())


min_num_nano = sorted(df["num_nano"].unique())[-2]
df = df[df["num_nano"] >= min_num_nano]
df = filter(df, best_nano=True)

best_nanos_sorted = df["name"].value_counts().index.tolist()
best_nano4s = [x for x in best_nanos_sorted if "M4" in x and "tuned" not in x]
best_nano8s = [x for x in best_nanos_sorted if "M8" in x and "tuned" not in x]

print("\n".join(best_nano4s[:5]))
print()
print("\n".join(best_nano4s[5:10]))
print()
print("\n".join(best_nano8s[:5]))
print()
print("\n".join(best_nano8s[5:10]))




