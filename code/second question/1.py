import pandas as pd
import numpy as np
from gurobipy import *

data = pd.read_csv(
    r"C:\Users\Administrator\Desktop\2023年MathorCup高校数学建模挑战赛赛题\A题\附件\附件1：data_100.csv").to_numpy().T
t = np.array([data[2 * i] for i in range(0, 100)])  # 通过率
h = np.array([data[2 * i + 1] for i in range(0, 100)])  # 坏账率
M = 100  # 信用评分卡数
N = 10  # 单张信用评分卡的阈值
P = 1000000
I = 0.08
c = np.max(t*h)


s = 0
f=np.empty(shape=1000000,dtype=float)

for i in range(M):
    for j in range(i+1,M):
        for k in range(j+1,M):
            for m in range(N):
                for n in range(N):
                    for o in range(N):
                        if 9*(h[i,m]+h[j,n]+h[k,o]) < 2:

                            s+=1
print(s)
