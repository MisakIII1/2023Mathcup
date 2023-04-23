from dwave.system import LeapHybridSampler
import pandas as pd
import numpy as np
from dimod import *
data = pd.read_csv(
    r"C:\Users\Administrator\Desktop\2023年MathorCup高校数学建模挑战赛赛题\A题\附件\data_processing.csv").to_numpy().T
t = np.array([data[2 * i] for i in range(0, 100)]) .flatten()   # 通过率
h = np.array([data[2 * i + 1] for i in range(0, 100)]).flatten()  # 坏账率
M = len(t)  # 信用评分卡数
P = 1000000
I = 0.08

x = np.array([Binary(f'x_{i}') for i in range(M)])
y = np.array([[Binary(f'y_{i}_{j}') for i in range(M)] for j in range(M)])

obj = P*quicksum(t[i]*t[j]*t[k]*y[i,j]*x[k]*(((1+I)/3)*(h[i]+h[j]+h[k])-I)
               for i in range(M) for j in range(i+1, M) for k in range(j+1, M))\
       +P*(quicksum(x)-3)**2+P*quicksum(3*y[i, j]+x[i]*x[j]-2*x[i]*y[i, j]-2*x[j]*y[i, j]
                                   for i in range(M) for j in range(i+1, M))

sampler = LeapHybridSampler(profile='gmail')
print(sampler.min_time_limit(obj))
samplest = sampler.sample(obj, time_limit=60)
print(-1*obj.energy(samplest.first.sample))
print(sorted(samplest.first.sample.items(), key=lambda x: x[1]))
print('采用量子退火算法求解近似方法的QUBO模型', '其目标函数为:', '\n', -1*obj.energy(samplest.first.sample), '\n', '其选择的信用评分卡和阈值为:')
for i in samplest.first.sample.items():
    if i[1] == 1 and i[0].startswith('x'):
        print(i)
