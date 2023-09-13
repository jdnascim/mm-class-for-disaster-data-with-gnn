import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import glob
import os

""" TABLES

    mobilenet - fusion and pca
    1: [4, 5, 6, 1, 2, 3]
    maxvit - fusion and pca
    1: [10, 11, 12, 7, 8, 9]
    gnnblocks
    1: [13, 9, 14]
"""
EXPS_NUM = [13, 9, 14]
SAMPLES_NUM = 10
DISCONSIDER = {}
RES = "./results/CrisisMMD/gnn/{}/{}_s{}.json"
X = [50, 100, 250, 500]

LABEL = [str(i) for i in EXPS_NUM]
TITLE = "Fusion and PCA"

results = np.zeros([len(EXPS_NUM), len(X)])

for exp_ix, exp in enumerate(EXPS_NUM):
    res_vec = np.zeros([len(X), SAMPLES_NUM])
    for i, s in enumerate(X):
        for j in range(SAMPLES_NUM):
            stats_file = RES.format(exp, s, j)

            print(stats_file)
            if os.path.isfile(stats_file):
                with open(stats_file, "r") as fp:
                    data = json.load(fp)
    
            res_vec[i][j] = np.round(data["f1_test"] * 100, 1) 

    res_vec[res_vec == 0] = np.nan
    f1 = np.nanmean(res_vec, axis=1)

    results[exp_ix] = np.round(f1, 1)

df = pd.DataFrame({'EXP':EXPS_NUM, str(X[0]): results[:,0], str(X[1]): results[:,1], str(X[2]): results[:,2], str(X[3]): results[:,3]})
df.to_csv("./results.csv", sep='\t')