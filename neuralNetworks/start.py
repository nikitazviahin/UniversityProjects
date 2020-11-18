import numpy as np


def act(x):
    return 0 if x < 0.5 else 1

def go(house, rock, attr):
    x = np.array([house, rock, attr])
    w11 = [0.3, 0.3, 0]
    w12 = [0.4, -0.5, 1]
    weight1 = np.array([w11, w12])
    weight2 = np.array([-1, 1])

    sum_hidden = np.dot(weight1,x)
    print('Hidden layer sum: '+str(sum_hidden))

    out_hidden = np.array([act(x) for x in sum_hidden])
    print('Out layer sum: '+str(out_hidden))

    sum_end = np.dot(weight2, out_hidden)
    y = act(sum_end)
    print("Output balue of NN: "+str(y))

    return y

house = 1
rock = 0
attr = 0

res = go(house, rock, attr)
if res == 1:
    print('I like you')
else:
    print('See you')
