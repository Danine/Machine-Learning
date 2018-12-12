import numpy as np
import matplotlib.pyplot as plt
from SMO import *
from Gaussiansmo import *

def kernel(x1, x2):
    return np.exp(-100 * (np.linalg.norm(x1 - x2) ** 2))

train_1 = np.loadtxt("C:/Users/Tyson/Desktop/original/data5/training_3.text")
alpha,x,y,b = Gaussiansmo(train_1[:,0:2],train_1[:,2],1,1000)
b = np.float(b)
temp = alpha*y
temp = np.transpose(temp)
omega = np.dot(temp,x)
omega = np.transpose(omega)

#画点
pos = []
neg = []
for i in range(len(train_1[:,0])):
	if train_1[i,2] == 1:
		pos.append(i)
	else:
		neg.append(i)
plt.scatter(train_1[pos,0], train_1[pos,1],marker='.')
plt.scatter(train_1[neg,0], train_1[neg,1],marker='x')
#画出边界上的支持向量
# xt = np.transpose(x)
# xx = np.transpose(xt)
# plt.scatter(xx[:,0], xx[:,1], marker='.')

# dataset = train_1[:,0:2]
# label = train_1[:,2]
S = (alpha > 1e-1).flatten()
alpha = alpha[S].flatten()
xm = np.array(x)[S]
ym = y[S]

xp = np.linspace(min(train_1[:,0]),max(train_1[:,0]))
yp = np.linspace(min(train_1[:,1]),max(train_1[:,1]))
X, Y = np.meshgrid(xp, yp)
vals = np.zeros((len(X), len(Y)))
for i in range(len(X)):
    for j in range(len(Y)):
        for k in range(len(alpha)):
            tmp = np.array([X[i, j], Y[i, j]])
            vals[i, j] -= alpha[k] * ym[k] * kernel(xm[k], tmp)
plt.contour(X, Y, vals, 0, linewidths = 1)
plt.show()