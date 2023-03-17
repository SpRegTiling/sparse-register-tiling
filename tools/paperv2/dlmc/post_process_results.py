import multiprocessing
import pandas as pd
import glob

from multiprocessing import Process
from artifact.utils import *
from tools.paper.plotting.plot_utils import filter

def thread_list():
    tmp_files = glob.glob(f"/tmp/figure7_to_9_pivoted/*.bcols*.threads*")
    thread_list = set()
    for file in tmp_files:
        thread_list.add(int(file.split("threads")[-1]))
    return list(thread_list)

def bcols_list():
    tmp_files = glob.glob(f"/tmp/figure7_to_9_pivoted/*.bcols*.threads*")
    thread_list = set()
    for file in tmp_files:
        thread_list.add(int(file.split("bcols")[-1].split(".")[0]))
    return list(thread_list)

def gen_post_processed_files():
    files = glob.glob(RESULTS_DIR + "/figure7_to_9_results*.csv")

    os.makedirs('/tmp/figure7_to_9_pivoted/', exist_ok=True)
    def pivot_process(file):
        df = pd.read_csv(file)
        df = post_process(df)
        for thread_count in df["numThreads"].unique():
            for bcols in df["n"].unique():
                dfw = pivot(df, bcols, thread_count)
                gend_file = f'/tmp/figure7_to_9_pivoted/{file.split("/")[-1]}.bcols{bcols}.threads{thread_count}'
                dfw.to_csv(gend_file)


    processes = [Process(target=pivot_process, args=(file,)) for file in files]
    for process in processes:
        process.start()

    for process in processes:
        process.join()

if __name__ == "__main__":
    gen_post_processed_files()