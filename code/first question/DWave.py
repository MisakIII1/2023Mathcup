import pandas as pd
import numpy as np
from dimod import *
from dwave.system import LeapHybridSampler

data = pd.read_csv(
    r"C:\Users\Administrator\Desktop\2023年MathorCup高校数学建模挑战赛赛题\A题\附件\附件1：data_100.csv").to_numpy().T
t = np.array([data[2 * i] for i in range(0, 100)])  # 通过率
h = np.array([data[2 * i + 1] for i in range(0, 100)])  # 坏账率
M = len(t)  # 信用评分卡数
N = len(t.T)  # 单张信用评分卡的阈值
P = 1000000
I = 0.08
c = np.max(t*h)

x = np.array([[Binary(f'x_{j}_{i}') for i in range(N)] for j in range(M)])
sampler = LeapHybridSampler(profile='163.com')

obj1 = quicksum(np.array([t[i, j] * x[i, j] - t[i, j] * h[i, j] * x[i, j] for i in range(M) for j in range(N)]))
obj2 = quicksum(np.array([t[i, j] * h[i, j] * x[i, j] for i in range(M) for j in range(N)]))
obj3 = (1-quicksum(quicksum(x)))**2
obj = P*obj2-P*I*obj1+P*c*obj3
samplest = sampler.sample(obj, time_limit=10)
print(samplest)
print(-1*obj.energy(samplest.first.sample))
print(sorted(samplest.first.sample.items(), key=lambda x: x[1]))
print('采用量子退火算法求解QUBO模型', '其目标函数为:', '\n', -P*obj.energy(samplest.first.sample), '\n', '其选择的信用评分卡和阈值为:')
for i in samplest.first.sample.items():
    if i[1] == 1 and i[0].startswith('x'):
        print(i)

