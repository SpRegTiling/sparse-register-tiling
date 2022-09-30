import os
import operator
import pandas as pd


def cache_miss_rate(data, layer, target):
    indices = data['name'] == target
    if layer == 1:
        miss_rate = data['PAPI_L1_DCM'][indices].to_numpy() / data["PAPI_LST_INS"][indices].to_numpy()
    elif layer == 2:
        miss_rate = data['PAPI_L2_DCM'][indices].to_numpy() / data['PAPI_L1_DCM'][indices].to_numpy()
    else:
        miss_rate = data["PAPI_L3_TCM"][indices].to_numpy() / data["PAPI_L3_TCA"][indices].to_numpy()
    return miss_rate


def cache_miss_cnt(data, layer, target):
    indices = data['name'] == target
    if layer == 1:
        miss_cnt = data['PAPI_L1_DCM'][indices].to_numpy()
    elif layer == 2:
        miss_cnt = data['PAPI_L2_DCM'][indices].to_numpy()
    else:
        miss_cnt = data["PAPI_L3_TCM"][indices].to_numpy()
    return miss_cnt


def total_memory_access(data, target):
    indices = data['name'] == target
    return data['PAPI_LST_INS'][indices].to_numpy()


def layer_memory_access(data, layer, target):
    indices = data['name'] == target
    if layer == 1:
        access_cnt = data["PAPI_LST_INS"][indices].to_numpy()
    elif layer == 2:
        access_cnt = data['PAPI_L1_DCM'][indices].to_numpy()
    else:
        access_cnt = data["PAPI_L3_TCA"][indices].to_numpy()
    return access_cnt


def get_experiment_results(directory, feature, target):
    result_vals = []
    result_labels = []
    for csv_file_name in os.listdir(directory):
        if csv_file_name.endswith('.csv'):
            data = pd.read_csv(directory + csv_file_name)
            indices = data['name'] == target
            result_vals.append(data[feature][indices].values)
            result_labels.append(int(csv_file_name[0: -4].split('-')[-1]))
    return zip(*sorted(zip(result_vals, result_labels), key=operator.itemgetter(1)))


if __name__ == '__main__':
    FOLDER_DIR = "../experiment_results/{}"
    file_name = "col_vector.csv"
    data = pd.read_csv(FOLDER_DIR.format(file_name))

    OPTIMIZED = "spmm_optimized<float>"
    MKL = "SpMMMKL<float>"

    experiment_dirs = {'row': 'row_experiment/', 'col': 'col_experiment/', 'random_sparse': 'random_sparse_experiment/'}

    target_list = [MKL, OPTIMIZED]

    for feature in data.columns:
        print(feature)
        val, label = get_experiment_results(FOLDER_DIR.format(experiment_dirs['col']), feature, OPTIMIZED)
        print(val)
    print(f"Unrolling Factors: {label}")

    # print("\nTotal Number of Memory Accesses:")
    # for target in target_list:
    #     print(f"{target}: \n{total_memory_access(data, target)} - Normalized by MKL: \t{total_memory_access(data, target) / total_memory_access(data, MKL)}")
    #
    # for layer in range(1, 4):
    #     print(f"\nL{layer} Cache Miss Rate")
    #     for target in target_list:
    #         print(f"{target}: \n{cache_miss_rate(data, layer, target)}")
    #
    # for layer in range(1, 4):
    #     print(f"\nL{layer} Cache Miss Number Normalized by MKL:")
    #     for target in target_list:
    #         print(f"{target}: \n{cache_miss_cnt(data, layer, target) / cache_miss_cnt(data, layer, MKL)}")
