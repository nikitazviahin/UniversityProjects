import pandas as pd
import numpy as np
import pylab


def func(z, x):
    return -z / x

try:
    a = float(input('Введите начальную границу промежутка: '))
    b = float(input('Введите конечную границу промежутка: '))
    if a>=b:
        print('Начальная граница не может быьб больше конечной')
        raise SystemExit
    elif a<-999999 or a>999999 or b>999999 or b<-999999:
        print('Интервал в не границ диапазона. Интервал должен находится в пределах -999999 до 999999')
        raise SystemExit
except:
    print('Введена строка, а не число')
    raise SystemExit
    
try:
    h = float(input('Введите шаг интервала: '))
    if h<=0 or h > 999999:
        print('Интервал должен быть между -999999 и 999999')
        raise SystemExit    
except:
    print('Введена строка, а не число')
    raise SystemExit

n = int((b-a)/(h/2))

matrix = [[1, -1, 1, None, None]]

for i in range(1, n + 1):
    matrix.append([None, None, None, None, None])
    k = []
    q = []
    matrix[i][0] = (matrix[i-1][0] + h/2)
    q.append(func(matrix[i-1][2], matrix[i-1][0]))
    q.append(func(matrix[i-1][2] + q[0]*h/3, matrix[i-1][0] + h/3))
    q.append(func(matrix[i-1][2] + q[1]*2*h/3, matrix[i-1][0] + 2*h/3))
    matrix[i][4] = q[0]*(4 + 3*q[2]/4)
    k.append(matrix[i-1][2])
    k.append(matrix[i-1][2] + q[0]*h/3)
    k.append(matrix[i-1][2] + q[1]*2*h/3)
    matrix[i][3] = k[0]*(4 + 3*k[2]/4)
    matrix[i][1] = (matrix[i-1][1] + matrix[i][3]*h)
    matrix[i][2] = (matrix[i-1][2] + matrix[i][4]*h)
matrix = pd.DataFrame(np.array(matrix),
                 columns=['x', 'y', 'z', 'k', 'q'])

print(matrix)
xlist = [i for i in matrix['x']]
ylist = [i for i in matrix['y']]
pylab.plot (xlist, ylist)

pylab.show()
