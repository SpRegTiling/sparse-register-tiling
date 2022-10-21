import altair as alt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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


df = pd.read_csv('/home/lwilkinson/Documents/layers.csv')

print(df[df["Name"] == "Dense"].iloc[0][1:].tolist()[:-2])

fig, axs = plt.subplots(3, 1)

sparse_linestyle = ':'
nano_linestyle = '--'
dense_linestyle = '-'

dense_color = 'black'
sparse_80_color = 'red'
sparse_70_color = 'blue'

maker = 'x'

Dense = np.array(df[df["Name"] == "Dense"].iloc[0][1:].tolist()[:-2], dtype=np.float32)
Sparse70 = np.array(df[df["Name"] == "Sparse70"].iloc[0][1:].tolist()[:-2], dtype=np.float32)
Sparse80 = np.array(df[df["Name"] == "Sparse80"].iloc[0][1:].tolist()[:-2], dtype=np.float32)
Nano70 = np.array(df[df["Name"] == "Nano70"].iloc[0][1:].tolist()[:-2], dtype=np.float32)
Nano80 = np.array(df[df["Name"] == "Nano80"].iloc[0][1:].tolist()[:-2], dtype=np.float32)

axs[0].plot(range(1, 29), Dense, marker='x', label="Dense", linestyle=dense_linestyle, color=dense_color)
axs[0].plot(range(1, 29), Sparse70, marker='x', label="Sparse70", linestyle=sparse_linestyle, color=sparse_70_color)
axs[0].plot(range(1, 29), Sparse80, marker='x', label="Sparse80", linestyle=sparse_linestyle, color=sparse_80_color)
axs[0].plot(range(1, 29), Nano70, marker='x', label="Nano70", linestyle=nano_linestyle, color=sparse_70_color)
axs[0].plot(range(1, 29), Nano80, marker='x', label="Nano80", linestyle=nano_linestyle, color=sparse_80_color)
axs[0].legend()

for i in range(5, 28, 2):
    # only one line may be specified; full height
    axs[0].axvline(x=i, color='b', label='axvline - full height', linewidth=0.5)


axs[1].plot(range(1, 29), np.cumsum(Dense), marker='x', label="Dense", linestyle=dense_linestyle, color=dense_color)
axs[1].plot(range(1, 29), np.cumsum(Sparse70), marker='x', label="Sparse70", linestyle=sparse_linestyle, color=sparse_70_color)
axs[1].plot(range(1, 29), np.cumsum(Sparse80), marker='x', label="Sparse80", linestyle=sparse_linestyle, color=sparse_80_color)
axs[1].plot(range(1, 29), np.cumsum(Nano70), marker='x', label="Nano70", linestyle=nano_linestyle, color=sparse_70_color)
axs[1].plot(range(1, 29), np.cumsum(Nano80), marker='x', label="Nano80", linestyle=nano_linestyle, color=sparse_80_color)


for i in range(5, 28, 2):
    # only one line may be specified; full height
    axs[1].axvline(x=i, color='b', label='axvline - full height', linewidth=0.5)

axs[1].axhline(y=df[df["Name"] == "Sparse70"].iloc[0][1:].tolist()[:-2][0], color='black', linestyle='-', linewidth=0.5)

print(np.divide(Dense, Nano70))
axs[2].plot(range(1, 29), list(np.divide(Dense, Sparse70)), marker='x', label="Sparse70", linestyle=sparse_linestyle, color=sparse_70_color)
axs[2].plot(range(1, 29), list(np.divide(Dense, Sparse80)), marker='x', label="Sparse80", linestyle=sparse_linestyle, color=sparse_80_color)
axs[2].plot(range(1, 29), list(np.divide(Dense, Nano70)), marker='x', label="Nano70", linestyle=nano_linestyle, color=sparse_70_color)
axs[2].plot(range(1, 29), list(np.divide(Dense, Nano80)), marker='x', label="Nano80", linestyle=nano_linestyle, color=sparse_80_color)

for i in range(5, 28, 2):
    # only one line may be specified; full height
    axs[2].axvline(x=i, color='b', label='axvline - full height', linewidth=0.5)

axs[2].axhline(y=1, color='black', linestyle='-', linewidth=0.5)
axs[2].set_ylim(0, 3)

plt.show()

