import pandas as pd
import numpy as np
from gurobipy import *

data = pd.read_csv(
    r"C:\Users\Administrator\Desktop\2023年MathorCup高校数学建模挑战赛赛题\A题\附件\附件1：data_100.csv").to_numpy().T
t = np.array([data[2 * i] for i in range(0, 100)])  # 通过率
h = np.array([data[2 * i + 1] for i in range(0, 100)])  # 坏账率
M = len(t)  # 信用评分卡数
N = len(t.T)  # 单张信用评分卡的阈值
P = 1000000
I = 0.08
c = np.max(t*h)

op_data = np.zeros(shape=100, dtype=str)
for i in range(M):
    model = Model('approximate_1')
    x = np.array([model.addVar(vtype=GRB.BINARY, name=f'x_{i+1}_{j+1}') for j in range(N)])
    model.addConstr(quicksum(x) == 1, name='选取每张信用卡的一个阈值')
    model.setObjective(P*quicksum(t[i, j]*x[j]*(h[i, j]-I*(1-h[i, j]))
                                        for j in range(N)), GRB.MINIMIZE)
    model.optimize()
    if model.status == GRB.OPTIMAL:
        for var in model.getVars():
            if var.x == 1.0 and var.varname.startswith('x'):
                op_data[i] = var.varname
                print(f'选择的第{i+1}个信用评分卡的阈值是：', var.varname)
        print('银行最终最优收益是：', -1*model.objVal)
