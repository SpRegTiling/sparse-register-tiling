import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Pivot benchmark output')
parser.add_argument('csv_file', help='CSV file to pivoted')

args = parser.parse_args()
df = pd.read_csv(args.csv_file)

df = df.pivot(index=['matrixPath', 'm', 'k', 'n', 'nnz'],
              columns='name',
              values=['correct', 'time median', 'time mean'])

filepath = "/".join(args.csv_file.split('.')[:-1])
df.to_csv(filepath + "_pivoted.csv")
