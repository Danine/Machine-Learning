import numpy as np
import matplotlib.pyplot as plt
from SMO import*

train_1 = np.loadtxt("training_2.txt")
alpha,x,y,b = SMO(train_1[:,0:2],train_1[:,2],1000)
b = np.float(b)
temp = alpha*y
temp = np.transpose(temp)
omega = np.dot(temp,x)

omega = np.transpose(omega)
test_1 = np.loadtxt("test_2.txt")
label = []
label = np.dot(test_1[:,0:2],omega)+b

#预测
for i in range(len(label)):
	if label[i] > 0:
		label[i] = 1
	elif label[i] < 0:
		label[i] = -1
	else:
		label[i] = 0
count = 0
for j in range(len(label)):
	if label[j] == test_1[j,2]:
		count = count + 1
	else:
		print(j)
result = count/len(label)
print("准确率：%s" %(result))

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
xt = np.transpose(x)
xx = np.transpose(xt)
plt.scatter(xx[:,0], xx[:,1], marker='.')
'''划线推导过程
XQ			  label						XQ		  -b
XQ x YP + b = label	→→→	label = 0  →→→	XQ x YP = -b  →→→	XY + QP = -b  →→→	求Q即yp
XQ			  label						XQ		  -b
X:test_1[:,0],Q:test_1[:,1],Y:omega[0],P:omega[1]
'''
xp = np.linspace(min(train_1[:,0]),max(train_1[:,0]))
yp = (-b-xp*omega[0])/omega[1]
plt.plot(xp, yp)

plt.show()