import gurobipy
import numpy as np
from gurobipy import *

model=Model('2')
x=np.array([model.addVar(vtype=GRB.BINARY,name=f'x{i}') for i in range(3)])
w=model.addVar(vtype=GRB.BINARY,name='w')
model.setObjective(w*(x[0]+x[1]+x[2]-1)+x[0]*x[1]+x[0]*x[2]+x[1]*x[2]-(x[0]+x[1]+x[2]-1),GRB.MINIMIZE)
model.Params.PoolSolutions=8
model.optimize()
if model.status==GRB.OPTIMAL:
    for v in model.getVars():
        print(v.varName,v.x)
else:print('无可行解')
