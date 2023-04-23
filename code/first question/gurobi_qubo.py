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

model = Model('first_qubo')
x = model.addVars(M, N, vtype=GRB.BINARY, name='x')
obj1 = QuadExpr(quicksum(t[i, j] * x[i, j] - t[i, j] * h[i, j] * x[i, j] for i in range(M) for j in range(N)))
obj2 = QuadExpr(quicksum(t[i, j]*h[i, j]*x[i, j] for i in range(M) for j in range(N)))
obj3 = QuadExpr((1-quicksum(x))**2)
model.setObjective(P*obj2-P*I*obj1+c*P*obj3, GRB.MINIMIZE)

model.optimize()
if model.status == GRB.OPTIMAL:
    for var in model.getVars():
        if var.x == 1.0:
            print('采用Gurobi求解第一问的QUBO模型','\n',
                  '选择的信用评分卡和阈值是：', var.varname,
                  '\n', '银行最终最优收益是：', -1*model.objVal)

# 选择的信用评分卡和阈值是： x[48,0]
# 银行最终最优收益是： 61172.0