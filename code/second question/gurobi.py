import pandas as pd
import numpy as np
from gurobipy import *

data = pd.read_csv(
    r"C:\Users\Administrator\Desktop\2023年MathorCup高校数学建模挑战赛赛题\A题\附件\附件1：data_100.csv").to_numpy().T
t = np.array([data[2 * i] for i in range(0, 3)])  # 通过率
h = np.array([data[2 * i + 1] for i in range(0, 3)])  # 坏账率
M = 3  # 信用评分卡数
N = 10  # 单张信用评分卡的阈值
P = 1000000
I = 0.08

model = Model('second')
x = np.array([[model.addVar(vtype=GRB.BINARY, name=f'x_{i}_{j}') for j in range(N)] for i in range(M)])
y = np.array([[[[model.addVar(vtype=GRB.BINARY, name=f'y_{i}_{j}_{m}_{n}') for n in range(N)]
                for m in range(N)] for j in range(M)] for i in range(M)])

obj = QuadExpr(quicksum(t[i, m]*t[j, n]*t[k, o]*y[i, j, m, n]*x[k, o]*(((1+I)/3)*(h[i, m]+h[j, n]+h[k, o])-I)
                         for i in range(M) for j in range(i+1, M) for k in range(j+1, M)
                         for m in range(N) for n in range(N) for o in range(N)))


model.addConstrs(quicksum(x[i]) == 1 for i in range(M))
model.addConstrs(x[i, m]*x[j, n]-y[i, j, m, n] == 0
                 for i in range(M) for j in range(i+1, M)
                 for m in range(N) for n in range(N))
model.setObjective(obj, GRB.MINIMIZE)

model.optimize()
if model.status == GRB.OPTIMAL:
    order = 0
    for var in model.getVars():
        if var.x == 1.0 and var.varname.startswith('x'):
            order += 1
            print(f'选择的第{order}个信用评分卡和阈值是：', var.varname)
    print('银行最终最优收益是：', -P * model.objVal )