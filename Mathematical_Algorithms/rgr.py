import pulp
 
 
 
 
# then, conduct initial declaration of problem
linearProblem = pulp.LpProblem("Maximizing for first objective",pulp.LpMaximize)
 
 
# delcare optimization variables, using PuLP
x1 = pulp.LpVariable("x1",lowBound = 0)
x2 = pulp.LpVariable("x2",lowBound = 0)
 
 
# add (first) objective function to the linear problem statement
linearProblem += 3*x1 + 3*x2
 
# add the constraints to the problem
linearProblem += 2*x1 + x2 <= 50
linearProblem += 2*x1 + x2 <= 5
 
# solve with default solver, maximizing the first objective
solution = linearProblem.solve()
 
# output information if optimum was found, what the maximal objective value is and what the optimal point is
print(str(pulp.LpStatus[solution])+" ; max value = "+str(pulp.value(linearProblem.objective))+
     " ; x1_opt = "+str(pulp.value(x1))+
     " ; x2_opt = "+str(pulp.value(x2)))
 
# remodel the problem statement
linearProblem = pulp.LpProblem("Maximize second objective",pulp.LpMaximize)
linearProblem += 4*x1 - 10*x2
linearProblem += x1 + x2 <= 10
linearProblem += 2*x1 + x2 <= 15
linearProblem += 2*x1 + 3*x2 >= 30
 
# review problem statement after remodelling
linearProblem
# apply default solver
solution = linearProblem.solve()
 
# output a string summarizing whether optimum was found, and if so what the optimal solution
print(str(pulp.LpStatus[solution])+" ; Максимальне значення цільової функції = "+str(pulp.value(linearProblem.objective))+
     " ; x1 оптимальне = "+str(pulp.value(x1))+
     " ; x2 оптимальне = "+str(pulp.value(x2)))
