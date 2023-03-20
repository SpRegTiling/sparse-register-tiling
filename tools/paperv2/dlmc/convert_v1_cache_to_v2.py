from tools.paperv2.utils import *
from tools.paperv2.dlmc.utils import *
from tools.paperv2.old_post_process import per_file_postprocess

import pandas as pd
import glob

def load_dlmc_df(subdir, nthreads=None, bcols=None):
    # assert nthreads or bcols

    if nthreads is not None and bcols is not None:
        return pd.read_csv(RESULTS_DIR + '/cache/' + subdir + f'/dlmc_bcols_{bcols}_nthreads_{nthreads}.csv')
    elif nthreads is not None:
        return pd.read_csv(RESULTS_DIR + '/cache/' + subdir + f'/dlmc_nthreads_{nthreads}.csv')
    elif bcols is not None:
        return pd.read_csv(RESULTS_DIR + '/cache/' + subdir + f'/dlmc_bcols_{bcols}.csv')
    else:
        all_files = glob.glob(os.path.join(RESULTS_DIR + 'cache/' + subdir, "*_per_part.csv" ))
        return pd.concat((pd.read_csv(f) for f in all_files), axis=0, ignore_index=True)



files = glob.glob(RESULTS_DIR + "/artifact/cascade/figure7_to_9_results*.csv")
dfs = [pd.read_csv(file) for file in files]
df = pd.concat(dfs)
df = post_process(df)
for numThreads in [1, 20]:
    for bcols in [32, 128, 256, 512]:
        out_file = cache_file_name("cascade" ,"all", bcols=bcols, threads=numThreads)
        dfw = pivot(df, drop_dupes=False, numThreads=numThreads, n=bcols)
        assert len(dfw)
        dfw.to_csv(out_file)
        print("Wrote:", out_file)

for numThreads in [1, 4]:
    df = pd.read_csv(RESULTS_DIR + f"cache/raspberrypi/dlmc_nthreads_{numThreads}.csv")
    df = df[(df.best_nano ==True) | (df.is_nano ==False)]
    dense_locs = df["name"].str.contains(r"MKL_Dense", regex=True)
    df.loc[dense_locs, "name"] = "MKL_Dense"
    df = post_process(df)
    
    for bcols in [32, 128, 512]:
        out_file = cache_file_name("raspberrypi" ,"all", bcols=bcols, threads=numThreads)
        dfw = pivot(df, drop_dupes=True, numThreads=numThreads, n=bcols)
        assert len(dfw)
        dfw.to_csv(out_file)
        print("Wrote:", out_file)
    
    for bcols in [256]:
        out_file = cache_file_name("raspberrypi" ,"all", bcols=bcols, threads=numThreads)
        df = pd.read_csv(RESULTS_DIR + f"/pi_rerun/pi_rerun_{bcols}_{numThreads}.csv")
        print(RESULTS_DIR + f"/pi_rerun/pi_rerun_{bcols}_{numThreads}.csv", len(df))
        df = per_file_postprocess(df, "pi")
        df = df[(df.best_nano ==True) | (df.is_nano ==False)]
        df = post_process(df)
    
        dfw = pivot(df, drop_dupes=True, numThreads=numThreads, n=bcols)
        assert len(dfw)
        dfw.to_csv(out_file)
        print("Wrote:", out_file)
        


# Use Rebutal Data

# df = pd.read_csv(RESULTS_DIR + "cache/cascadelake/dlmc_nthreads_1.csv")
# dense_locs = df["name"].str.contains(r"MKL_Dense", regex=True)
# df.loc[dense_locs, "name"] = "MKL_Dense"
# df = post_process(df)
# for numThreads in [1]:
#     for bcols in [32, 128, 256, 512]:
#         out_file = cache_file_name("cascade" ,"all", bcols=bcols, threads=numThreads)
#         dfw = pivot(df, drop_dupes=True, numThreads=numThreads, n=bcols)
#         assert len(dfw)
#         dfw.to_csv(out_file)
#         print("Wrote:", out_file)

# df = pd.read_csv(RESULTS_DIR + "cache/cascadelake/rebuttal_20/dlmc_nthreads_20.csv")
# dense_locs = df["name"].str.contains(r"MKL_Dense", regex=True)
# df.loc[dense_locs, "name"] = "MKL_Dense"
# df = post_process(df)
# for numThreads in [20]:
#     for bcols in [32, 128, 256, 512]:
#         out_file = cache_file_name("cascade" ,"all", bcols=bcols, threads=numThreads)
#         dfw = pivot(df, drop_dupes=True, numThreads=numThreads, n=bcols)
#         assert len(dfw)
#         dfw.to_csv(out_file)
#         print("Wrote:", out_file)