def add_collection_column(df):
    def map_to_collection(matrix_path):
        collection_mapping = {
            "ss_mtx_collection": "SuiteSparse",
            "dlmc": "DLMC"
        }

        for k, v in collection_mapping.items():
            if k in matrix_path: return v
        return "Unknown"

    df["Collection"] = df["matrixPath"].map(map_to_collection)
    return df


def add_matrix_id_column(df):
    df = add_collection_column(df)
    # Construct a unique matrixId that is independent of dataset directory
    df["matrixId"] = df["Collection"] + "|" + df["matrixPath"].str \
        .split(r'ss_mtx_collection|dlmc', regex=True).str[-1]
    return df


def remove_tuned_results(df):
    return df[~df["tuning"].str.contains("\|", na=False)]


def tuning_results(df):
    return df.copy()[~df["tuning"].str.contains("saved result", na=True)]


def compute_pruning_method_and_model(df):
    # Construct a unique matrixId that is independent of dataset directory
    df["pruningMethod"] = df["matrixPath"].str.split("/").str[-3]
    df["model"] = df["matrixPath"].str.split("/").str[-4]
    df["sparsityFolder"] = df["matrixPath"].str.split("/").str[-2]
    df["sparsityFolder"] = round(df["sparsityFolder"].astype(float), 2)
    return df


def compute_matrix_properties(df, sparsity_round=True):
    if sparsity_round:
        df["sparsity"] = round(1 - df["nnz"] / (df["m"] * df["k"]), 2)
    else:
        df["sparsity"] = 1 - df["nnz"] / (df["m"] * df["k"])

    df["aspect ratio"] = round(df["m"] / df["k"], 2)
    df["aspect ratio +/-"] = df[['m', 'k']].max(axis=1) / df[['m', 'k']].min(axis=1)

    df.loc[df["m"] < df["k"], "aspect ratio +/-"] *= -1
    return df


def compute_speed_up_vs(df, name, group_by=["matrixPath", "n"]):
    def compute_time_vs(x):
        baseline = x[x["name"] == name].iloc[0]["time median"]
        x[f'Speed-up vs {name}'] = baseline / x["time median"]
        return x

    df = df.groupby(group_by).apply(compute_time_vs).reset_index(drop=True)
    return df


def compute_scaling(df, group_by=["matrixPath", "n", "name"]):
    def compute_time_vs(x):
        baseline = x[x["numThreads"] == 1].iloc[0]["time median"]
        x[f'scaling'] = baseline / x["time median"]
        return x

    df = df.groupby(group_by).apply(compute_time_vs).reset_index(drop=True)
    return df


def compute_best(df, group_by=["matrixPath", "n", "numThreads", "name"]):
    def compute_best(x):
        x["best"] = False
        x.loc[x["time median"].idxmin(), "best"] = True
        return x


    return df.groupby(group_by).apply(compute_best).reset_index(drop=True)


def compute_global_best(df, group_by=["matrixPath", "n", "numThreads"]):
    def compute_best(x):
        x["global_best"] = False
        x.loc[x["time median"].idxmin(), "global_best"] = True
        return x


    return df.groupby(group_by).apply(compute_best).reset_index(drop=True)


def compute_avg_speedup_per_config(df,
                                   speed_up_vs='MKL_Executor_Only',
                                   group_by=["name", "n", "m_tile", "k_tile", "n_tile"]):

    return df.groupby(group_by, dropna=False).mean().reset_index()


def compute_max_speedup_per_config(df,
                                   speed_up_vs='MKL_Executor_Only',
                                   group_by=["name", "n", "m_tile", "k_tile", "n_tile"]):

    def compute_fastest(x):
        fastest_time = 1e12
        fastest_method = ""

        for index, row in x.iterrows():
            if row["time median"] < fastest_time:
                fastest_time = row["time median"]
                fastest_method = row["name"]
        x["fastest_config"] = fastest_method

        return x


    return df.groupby(group_by, dropna=False).apply(compute_fastest).reset_index(drop=True)