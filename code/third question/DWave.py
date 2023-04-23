import pandas as pd
import numpy as np
from dimod import *
from dwave.system import LeapHybridSampler

data = pd.read_csv(
    r"C:\Users\Administrator\Desktop\2023年MathorCup高校数学建模挑战赛赛题\A题\附件\附件1：data_100.csv").to_numpy().T
t = np.array([data[2 * i] for i in range(0, 50)])  # 通过率
h = np.array([data[2 * i + 1] for i in range(0, 50)])  # 坏账率
M = 10  # 信用评分卡数
N = 10  # 单张信用评分卡的阈值
P = 1000000
I = 0.08
c = np.max(t*h)

x = np.array([[Binary(f'x_{j}_{i}') for i in range(N)] for j in range(M)])
y = np.array([Binary(f'y_{i}') for i in range(M)])
z = np.array([[[[Binary(f'z_{i}_{j}_{m}_{n}') for n in range(N)]
                for m in range(N)] for j in range(M)] for i in range(M)])


obj = quicksum(t[i, m]*t[j, n]*t[k, o]*z[i, j, m, n]*x[k, o]*(((1+I)/3)*(h[i, m]+h[j, n]+h[k, o])-I)
               for i in range(M) for j in range(i+1, M) for k in range(j+1, M)
               for m in range(N) for n in range(N) for o in range(N))+\
      (quicksum((quicksum(x[i])-y[i])**2 for i in range(M))+(quicksum(y)-3)**2+quicksum(3*z[i, j, m, n]+x[i, m]*x[j, n]-2*x[i, m]*z[i, j, m, n]-2*x[j, n]*z[i, j, m, n]
               for i in range(M) for j in range(i+1, M)
               for m in range(N) for n in range(N)))

sampler = LeapHybridSampler(profile='gmail')
print(sampler.min_time_limit(obj))
samplest = sampler.sample(obj,time_limit=120)
print(samplest)
print(-P*obj.energy(samplest.first.sample))
print(sorted(samplest.first.sample.items(), key=lambda x: x[1]))