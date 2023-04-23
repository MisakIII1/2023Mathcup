import pandas as pd
import numpy as np
from gurobipy import *

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

model = Model('third_finally')
x = model.addVars(M, N, vtype=GRB.BINARY, name='x')
y = model.addVars(M, vtype=GRB.BINARY, name='y')
w = model.addVars(M, M, M, N, N, N, vtype=GRB.BINARY,name='w')

obj = QuadExpr()
for i in range(M):
    for j in range(i+1, M):
        for k in range(j+1, M):
            for m in range(N):
                for n in range(N):
                    for o in range(N):
                        if 9*(h[i, m]+h[j, n]+h[k, o]) < 2:
                            obj.add(w[i, j, k, m, n, o]*(x[i, m]+x[j, n]+x[k, o]-2),
                                    t[i, m]*t[j, n]*t[k, o]*(f1*(h[i, m]+h[j, n]+h[k, o])-I)
                                    )


model.addConstrs(quicksum(x[i]) == y[i] for i in range(M))
model.addConstr(quicksum(y) == 3, name='最多选取3个')


model.setObjective(obj, GRB.MINIMIZE)
model.optimize()
if model.status == GRB.OPTIMAL:
    order = 0
    for var in model.getVars():
        if var.x == 1.0 and var.varname.startswith('x'):
            order += 1
            print(f'选择的第{order}个信用评分卡和阈值是：', var.varname)
    print('银行最终最优收益是：', -P * model.objVal)