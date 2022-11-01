import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import torch

from padding_gemm_bench import gemm, spmm

plot_gemm = False # make this true if you want to plot gemm heatmaps

def get_max_min(lst, label):
    max = .0; min=10000.0
    for l in lst:
        max = np.max ((np.max(l[label].values), max))
        min = np.min ((np.min(l[label].values), min))
    max = max + 0.15*max if max>1 else max*2
    min = min - 0.15 * min if min > 1 else min / 2
    return max, min


plt.rcParams["figure.figsize"] = (15,7)
file_name = "results" \
            "/hybrid_70_80_agg_MKL_resnet.csv"
df = pd.read_csv(file_name)

### GEMM heatmaps
def timing_sparse_dense_MM(m, n, bcol, sr):
    A_spm_mat = scipy.sparse.random(m, n, 1-sr, dtype=np.float32).toarray()
    torch_csr = torch.tensor(A_spm_mat).to_sparse_csr()
    A_dense_mat = torch_csr.to_dense()
    Bmat = torch.ones(n, bcol, dtype=torch.float32)
    num_test = 9
    t_array = []
    for i in range(num_test):
        C_gemm, tg = gemm(A_dense_mat, Bmat)
        t_array.append(tg)
    t_gemm = np.median(t_array)
    C_gemm, tg = gemm(A_dense_mat, Bmat)
    t_array = []
    for i in range(num_test):
        C_spmm, ts = spmm(torch_csr, Bmat)
        t_array.append(ts)
        cmp = torch.eq(C_spmm, C_gemm)
        #if not C_gemm.shape[0] * C_gemm.shape[1] == cmp.sum().sum():
        #    print(" SpMM NOT EQUAL! ")
    t_spmm = np.median(t_array)
    return t_gemm, t_spmm


def plot_estimate(sp_ratio):
    density = 1 - sp_ratio
    dim = np.power(2, np.arange(1, 12, 1))
    bcol_dim = np.power(2, np.arange(1, 10, 1))
    dense_plot = np.zeros((len(dim), len(bcol_dim)))
    sparse_plot = np.zeros((len(dim), len(bcol_dim)))
    dense_mkl = np.zeros((len(dim), len(bcol_dim)))
    sparse_mkl = np.zeros((len(dim), len(bcol_dim)))
    for i in range(len(dim)):
        for j in range(len(bcol_dim)):
            dense_flop = 0#dim[i]*dim[i]*bcol_dim[j]
            sparse_flop = 0#density*(dim[i]*dim[i]) * bcol_dim[j]
            dense_plot[i, j] = dim[i] * bcol_dim[j] + dim[i] * (dim[i] * dim[i] +
                                                                dim[i] *
                                                                bcol_dim[j]) + dense_flop
            sparse_plot[i, j] = dim[i] * bcol_dim[j] + dim[i] * ((2*density *
            dim[i] * dim[i] + dim[i]) + density * (dim[i] * bcol_dim[j])) \
                                + sparse_flop
            [dense_mkl[i, j], sparse_mkl[i, j]] = timing_sparse_dense_MM(dim[i],
                                                                 dim[i], bcol_dim[j], sp_ratio)
    return dense_plot, sparse_plot, dim, bcol_dim, dense_mkl, sparse_mkl

if plot_gemm:
    pp = plt.subplots(2, 7, figsize=(22,8))
    [im1, im2] = [None, None]
    for idx, item in enumerate( (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99) ):
        [dp, sp, dim, bcols, mkl_d, mkl_s] = plot_estimate(item)
        im1 = pp[1][0, idx].imshow(sp/dp, cmap='hot',
                         interpolation='none', origin='lower')
        im2 = pp[1][1, idx].imshow(mkl_s / mkl_d, cmap='hot',
                             interpolation='none', origin='lower')
        pp[1][0, idx].set_xticks(np.arange(len(bcols)))
        pp[1][0, idx].set_xticklabels([str(x) for x in bcols])
        pp[1][1, idx].set_xticks(np.arange(len(bcols)))
        pp[1][1, idx].set_xticklabels([str(x) for x in bcols])
        pp[1][0, idx].set_yticks(np.arange(len(dim)))
        pp[1][0, idx].set_yticklabels(dim)
        pp[1][1, idx].set_yticks(np.arange(len(dim)))
        pp[1][1, idx].set_yticklabels(dim)
        pp[1][0, idx].set_title("sp ratio:" + str(item))
        pp[1][0, idx].set_aspect(1.3)
    #    pp[1][idx].set_xticklabels(x_label_list)
    pp[1][0, 0].set_ylabel("Dim m")
    pp[1][0, 0].set_xlabel("Number of columns in B")
    pp[0].colorbar(im1, ax=pp[1][0, 0])
    pp[0].colorbar(im2, ax=pp[1][1, 0])
    plt.show()
    quit()



############### Plotting speedup figures
names = df.matrixPath.unique()
df_cherry = []
for bcol in (64, 128, 256): # update this to reflect what is needed
    cherry = []
    for name in names:
        df_test = df.loc[df["matrixPath"] == name]
        df_test2 = df_test.loc[df["B Col"] == bcol]
        df_test2.loc[df_test2["matrixPath"] == name, 'Hybrid Sep Speedup'] = \
            np.max(df_test2["PyTorch GEMM Time (sec)"] / df_test2["Total Hybrid Sep Time (sec)"] )
        df_test2.loc[df_test2["matrixPath"] == name, 'Hybrid Blocked Speedup']\
            = np.max( df_test2["PyTorch GEMM Time (sec)"] / df_test2['Total ' \
                                                                 'Hybrid ' \
                                                             'Blocked Time ('
                                                                     'sec)'])
        df_test2.loc[df_test2["matrixPath"] == name, 'Hybrid Full-Blocked Speedup'] \
            = np.max( df_test2["PyTorch GEMM Time (sec)"] / df_test2[
            'Total Hybrid Full-Blocked Time (sec)'])

        df_test2.loc[df_test2["matrixPath"] == name, 'Hybrid Speedup'] = \
            np.max([np.max(df_test2.loc[df_test2["matrixPath"] == name,
                                        'Hybrid Blocked Speedup']), np.max(df_test2.loc[df_test2["matrixPath"] == name,
                                        'Hybrid Sep Speedup']), np.max(df_test2.loc[df_test2["matrixPath"] == name,
                                        'Hybrid Full-Blocked Speedup'])])

        cherry.append(df_test2.loc[df_test2['Hybrid Speedup'].idxmax()])
    df_cherry.append(pd.DataFrame(cherry))

HYB_FB='Hybrid Full-Blocked Speedup'
HYB_SEP='Hybrid Sep Speedup'
HYB_B='Hybrid Blocked Speedup'
GEMM = 'PyTorch GEMM Time (sec)'
#fig, ax1 = plt.subplots()
colors = ["black", "red", "blue",  "purple", "yellow", "green"]
t = len(df_cherry)
tp = t
pp = plt.subplots(1, tp)
xaxis = np.arange(len(df_cherry[0]['matrixPath'].values))
max_v, min_v = get_max_min(df_cherry, "Hybrid Speedup")
for ii in range(t):
    pp[1][ii].set_title('Bcol '+ str(df_cherry[ii]['B Col'].values[0]))
    pp[1][ii].scatter(xaxis, df_cherry[ii]["Hybrid Speedup"].values,
            color=colors[0],marker='^')
    # pp[1][ii].scatter(xaxis, df_cherry[ii][HYB_B].values,
    #                   color=colors[1])
    # pp[1][ii].scatter(xaxis, df_cherry[ii][HYB_FB].values,
    #                   color=colors[2])
    # pp[1][ii].scatter(xaxis, df_cherry[ii][HYB_SEP].values,
    #                   color=colors[3])
    pp[1][ii].set_ylim([min_v, max_v])
    pp[1][ii].set_xlabel("Matrix ID")
pp[1][0].set_ylabel("Hybrid Speedup over PyTorch GEMM")
plt.show()


def plot_spmm(pp):
    xaxis = np.arange(len(df_cherry[0]['matrixPath'].values))
    for ii in range(t):
        pp[1][ii].set_title('Bcol ' + str(df_cherry[ii]['B Col'].values[0]))
        spmm_su = df_cherry[ii][GEMM].values / df_cherry[ii]["PyTorch SpMM Time (sec)"].values
        pp[1][ii].scatter(xaxis, spmm_su, color=colors[ii])
        pp[1][ii].set_ylim([0.4, 2])
        pp[1][ii].set_xlabel("Matrix ID")
    pp[1][0].set_ylabel("PyTorch SpMM peedup over PyTorch GEMM")
    plt.show()
spmm_plot = plt.subplots(1, tp)
plot_spmm(spmm_plot)


def plot_best(pp):
    xaxis = np.arange(len(df_cherry[0]['matrixPath'].values))
    for ii in range(t):
        pp[1][ii].set_title('Bcol ' + str(df_cherry[ii]['B Col'].values[0]))
        spmm_su = df_cherry[ii][GEMM].values / df_cherry[ii]["PyTorch SpMM Time (sec)"].values
        best_su = df_cherry[ii]['Hybrid Speedup'].values / spmm_su
        best_su = np.minimum(df_cherry[ii]['Hybrid Speedup'].values, best_su)
        print(df_cherry[ii]['B Col'].values[0], " : ", np.average(best_su),
              " Max: ", np.min(best_su) )
        pp[1][ii].scatter(xaxis, best_su, color=colors[ii])
        pp[1][ii].set_ylim([0.4, 3])
        pp[1][ii].set_xlabel("Matrix ID")
    pp[1][0].set_ylabel("Best Hybrif Speedup over Bedt of PyTorch Sparse/Dense")
    plt.show()
best_plot = plt.subplots(1, tp)
plot_best(best_plot)


pp = plt.subplots(1, tp)
max_v, min_v = get_max_min(df_cherry, "Hybrid Speedup")
for ii in range(t):
    pp[1][ii].set_title('Bcol '+ str(df_cherry[ii]['B Col'].values[0]))
    xaxis = df_cherry[ii]["dense computation ratio"].values
    #xaxis = 1.0/df_cherry[ii]['hybrid padding'].values * df_cherry[ii][
    # "dense computation ratio"].values
    pp[1][ii].scatter(xaxis, df_cherry[ii]["Hybrid Speedup"].values,
            color=colors[ii])
    pp[1][ii].set_ylim([min_v, max_v])
    pp[1][ii].set_xlabel("Dense/ Sparse Ratio")
pp[1][0].set_ylabel("Hybrid Speedup over MKL GEMM")
plt.show()


def plot_padded(pp):
    xaxis = np.arange(len(df_cherry[0]['matrixPath'].values))
    for ii in range(t):
        pp[1][ii].set_title('Bcol '+ str(df_cherry[ii]['B Col'].values[0]))
        ratio = df_cherry[ii]['hybrid padding'].values/df_cherry[ii]['nnz'].values
        pp[1][ii].scatter(xaxis, ratio, color=colors[ii])
        gemm_pad = df_cherry[ii]['m'].values*df_cherry[ii]['k'].values - \
                   df_cherry[ii]['nnz'].values
        gemm_pad = gemm_pad / df_cherry[ii]['nnz'].values
        pp[1][ii].scatter(xaxis, gemm_pad, color='green', label="GEMM Padding")
        pp[1][ii].set_ylim([0, 2.6])
        pp[1][ii].set_ylabel("Number of Padded NNZ / NNZ")
        pp[1][ii].set_xlabel("Matrix ID")
    plt.legend()
    plt.show()
padded = plt.subplots(1, tp)
plot_padded(padded)


def plot_tdense(pp):
    xaxis = np.arange(len(df_cherry[0]['matrixPath'].values))
    max_v, min_v = get_max_min(df_cherry, "number of dense tiles")
    for ii in range(t):
        pp[1][ii].set_title('Bcol '+ str(df_cherry[ii]['B Col'].values[0]))
        ratio = df_cherry[ii]['number of dense tiles'].values
        pp[1][ii].scatter(xaxis, ratio, color=colors[ii])
        #pp[1][ii].set_ylim([min_v, max_v])
        pp[1][ii].set_ylabel("Number of Dense Tiles")
        pp[1][ii].set_xlabel("Matrix ID")
    plt.show()
p_dense = plt.subplots(1, tp)
plot_tdense(p_dense)


def plot_sparse_dense(pp):
    xaxis = np.arange(len(df_cherry[0]['matrixPath'].values))
    for ii in range(t):
        pp[1][ii].set_title('Bcol ' + str(df_cherry[ii]['B Col'].values[0]))
        dt = df_cherry[ii]["Hybrid Dense Sep Time (sec)"].values
        st = df_cherry[ii]["Hybrid Sparse Sep Time (sec)"].values
        tot = dt + st
        pp[1][ii].bar(xaxis, dt/tot, color=colors[0], label="Dense Time Ratio")
        pp[1][ii].bar(xaxis, st / tot, bottom=dt/tot, color=colors[1],
                      label = "Sparse Time Ratio")
        #pp[1][ii].set_ylim([0.75, 2])
        pp[1][ii].set_xlabel("Matrix ID")
    pp[1][0].set_ylabel("Time / Total Hybrid Time")
    plt.legend()
    plt.show()
spd_plot = plt.subplots(1, tp)
plot_sparse_dense(spd_plot)


# `density threshold`
def plot_density_threshold(pp):
    xaxis = np.arange(len(df_cherry[0]['matrixPath'].values))
    for ii in range(t):
        pp[1][ii].set_title('Bcol ' + str(df_cherry[ii]['B Col'].values[0]))
        dt = df_cherry[ii]['density threshold'].values
        pp[1][ii].bar(xaxis, dt, color=colors[0], label="Dense Time Ratio")
        #pp[1][ii].set_ylim([0.75, 2])
        pp[1][ii].set_xlabel("Matrix ID")
    pp[1][0].set_ylabel("density threshold")
    plt.legend()
    plt.show()
dt_plot = plt.subplots(1, tp)
plot_density_threshold(dt_plot)
