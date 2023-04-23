import pandas as pd
import numpy as np
from gurobipy import *

data = pd.read_csv(
    r"C:\Users\Administrator\Desktop\2023年MathorCup高校数学建模挑战赛赛题\A题\附件\data_processing.csv").to_numpy().T
t = np.array([data[2 * i] for i in range(0, 100)]) .flatten()   # 通过率
h = np.array([data[2 * i + 1] for i in range(0, 100)]).flatten()  # 坏账率
M = len(t)  # 信用评分卡数
P = 1000000
I = 0.08
c = np.max(t*h)

model = Model('gurobi')
x = np.array([model.addVar(vtype=GRB.BINARY, name=f'x_{i}') for i in range(M)])
y = np.array([[model.addVar(vtype=GRB.BINARY, name=f'y_{i}_{j}') for j in range(M)] for i in range(M)])
model.addConstr(quicksum(x) == 3, name='选择且只选择三个信用评分卡')
model.addConstrs(y[i, j]-x[i]*x[j] == 0 for i in range(M) for j in range(i+1, M))

model.setObjective(quicksum(t[i]*t[j]*t[k]*y[i, j]*x[k]*(((I+1)/3)*(h[i]+h[j]+h[k])-I)
                            for i in range(M) for j in range(i+1, M) for k in range(j+1, M)), GRB.MINIMIZE)

model.optimize()
if model.status == GRB.OPTIMAL:
    order = 0
    for var in model.getVars():
        if var.x == 1.0 and var.varname.startswith('x'):
            order += 1
            print(f'选择的第{order}个信用评分卡和阈值是：', var.varname)
    print('银行最终最优收益是：', -P * model.objVal)