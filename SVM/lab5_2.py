import numpy as np
import matplotlib.pyplot as plt
from SMO import *

datafile = open('train-01-images.svm', mode='r')
trdata = datafile.readlines()
datafile.close()
trainingdata = []
for i in trdata:
    trainingdata.append(i.replace('\n', '').split(' ', i.count(' ')))
dataset = []
label = []
r = 100/255
for i in trainingdata:
    i.pop()
    label.append(int(i[0]))
    i.pop(0)
    tmp = []
    for j in range(784):
        tmp.append(0)
    for j in i:
        plz = j.find(':')
        tmp[int(j[0:plz])] = int(j[plz+1:])*r
    dataset.append(tmp)

dataset = np.array(dataset)
label = np.array(label)

# ytmp = label.reshape(-1, 1) * 1.
# xtmp = ytmp * dataset

alpha,x,y,b = SMO(dataset,label,10)
b = np.float(b)
temp = alpha*y
temp = np.transpose(temp)
omega = np.dot(temp,x)

# omega = np.transpose(omega)

datafile = open('test-01-images.svm', mode='r')
tedata = datafile.readlines()
datafile.close()
testdata = []
for i in tedata:
    testdata.append(i.replace('\n', '').split(' ', i.count(' ')))
dataset = []
label = []
r = 100/255
for i in testdata:
    i.pop()
    label.append(int(i[0]))
    i.pop(0)
    tmp = []
    for j in range(784):
        tmp.append(0)
    for j in i:
        plz = j.find(':')
        tmp[int(j[0:plz])] = int(j[plz+1:])*r
    dataset.append(tmp)

dataset = np.array(dataset)
label = np.array(label)
target = []
target = np.dot(dataset,omega)+b
for i in range(len(target)):
	if target[i] > 0:
		target[i] = 1
	elif target[i] < 0:
		target[i] = -1
	else:
		target[i] = 0
count = 0
for i in range(len(label)):
    if label[i] == target[i]:
        count += 1
    else:
        print(i)
print("训练值数量：%s" %(len(trdata)))
print("准确率：%s" %(count / len(label)))