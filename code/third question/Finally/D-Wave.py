import pandas as pd
import numpy as np
from dimod import *
from dwave.system import LeapHybridSampler

data = pd.read_csv(
    r"C:\Users\Administrator\Desktop\2023年MathorCup高校数学建模挑战赛赛题\A题\附件\附件1：data_100.csv").to_numpy().T
t = np.array([data[2 * i] for i in range(0, 100)])  # 通过率
h = np.array([data[2 * i + 1] for i in range(0, 100)])  # 坏账率
M = 100  # 信用评分卡数
N = len(t.T)  # 单张信用评分卡的阈值
P = 1000000
I = 0.08
f1 = 0.36
c = np.max(t*h)

model = BQM('BINARY')
x = np.array([[Binary(f'x_{j}_{i}') for i in range(N)] for j in range(M)])
y = np.array([Binary(f'y_{i}') for i in range(M)])
w = np.array([[[[[[Binary(f'w_{i}_{j}_{k}_{m}_{n}_{o}')
                for o in range(N)] for n in range(N)] for m in range(N)]
                for k in range(M)] for j in range(M)] for i in range(M)])
obj = BQM('BINARY')

for i in range(M):
    for j in range(i+1, M):
        for k in range(j+1, M):
            for m in range(N):
                for n in range(N):
                    for o in range(N):
                        if 9*(h[i, m]+h[j, n]+h[k, o]) < 2:
                            obj = quicksum([obj, P*(w[i, j, k, m, n, o]*(x[i, m]+x[j, n]+x[k, o]-2))
                                            *t[i, m]*t[j, n]*t[k, o]*(f1*(h[i, m]+h[j, n]+h[k, o])-I)])






# sampler=LeapHybridSampler(profile='gmail')
# samplest = sampler.sample(obj, time_limit=30)
# print(samplest)
# print(-1*obj.energy(samplest.first.sample))
# print(sorted(samplest.first.sample.items(), key=lambda x: x[1]))
# print('采用量子退火算法求解QUBO模型', '其目标函数为:', '\n', -1*obj.energy(samplest.first.sample), '\n', '其选择的信用评分卡和阈值为:')
# for i in samplest.first.sample.items():
#     if i[1] == 1 and i[0].startswith('x'):
#         print(i)