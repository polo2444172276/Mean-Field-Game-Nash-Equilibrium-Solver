import numpy as np
x = [-0.5434, 1.73285]
y, l1, l2 = [], [], []
f = lambda x: 35*x/(4*x+6)
l1_f = lambda x: (6-x)/x
l2_f = lambda x: (35-5*x)/x
for i in x:
    y.append(f(i))
    l1.append(l1_f(i))
    l2.append(l2_f(f(i)))
print(y,l1,l2)
