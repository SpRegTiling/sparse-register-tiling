import glob
import sys
import matplotlib.pyplot as plt
import numpy as np
from tools.paper.plotting.plot_utils import plot_save
from matplotlib import rc, rcParams
from tools.paper.plotting.plot_utils import *
from scipy import stats

from artifact.utils import *
from artifact.figure7_to_9.post_process_results import get_df, thread_list, bcols_list


sparsitySamePerLayerMethods = set()
sparsityNotSamePerLayerMethods = set()


for thread_count in [1,20]:
    dfs = []
    for bcol in bcols_list():
        df = get_df(bcol, thread_count)
        df["sparsitySamePerLayer"] = False
        for prungingMethod in df["pruningMethod"]:
            matching_rows = df["pruningMethod"] == prungingMethod
            random_target = df[matching_rows].iloc[0]["pruningModelTargetSparsity"]
            if df[matching_rows & (df["pruningModelTargetSparsity"] == random_target)]["sparsity"].std() < 1e-14:
                sparsitySamePerLayer = True
                sparsitySamePerLayerMethods.add(prungingMethod)
            else:
                sparsitySamePerLayer = False
                sparsityNotSamePerLayerMethods.add(prungingMethod)
            df.loc[matching_rows, "sparsitySamePerLayer"] = sparsitySamePerLayer
        dfs.append(df)


        print(len(df))
        df.to_csv(f"results_nthreads_{thread_count}_bcols_{bcol}.csv")
    df = pd.concat(dfs)
    df.to_csv(f"results_nthreads_{thread_count}_bcols_all.csv")

    print(df["pruningMethod"].unique())

print("sparsitySamePerLayerMethods", sparsitySamePerLayerMethods)
print("sparsityNotSamePerLayerMethods", sparsityNotSamePerLayerMethods)