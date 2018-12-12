'''
iter:迭代次数
nu:SVM对应类型的参数
obj:SVM文件转换为的二次规划求解得到的最小值
rho:判决函数的常数项b
nSV:支持向量个数, nBSV:边界上的支持向量个数
Total nSV:支持向量总个数
Accuracy：测试准确率
'''
import numpy as np
import matplotlib.pyplot as plt
import SMO
from libsvm.python.svmutil import *
from libsvm.python.svm import *

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

model = svm_train(label, dataset)

datafile = open('train-01-images.svm', mode='r')
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

xt = np.array(dataset)
yt = np.array(label)

p_label, p_acc, p_val = svm_predict(yt, xt, model)




# pos = []
# neg = []
# for i in range(len(train_1[:,0])):
# 	if train_1[i,0] == 1:
# 		pos.append(i)
# 	else:
# 		neg.append(i)

# plt.scatter(train_1[pos,1], train_1[pos,2],marker='.')
# plt.scatter(train_1[neg,1], train_1[neg,2],marker='x')
# plt.show()



# train_sigma = np.std(train_1,axis=0,ddof=1)#标准差
# train_mu = np.mean(train_1,axis=0)#求平均值
# train_1[:,0] = (train_1[:,0] - train_mu[0])/train_sigma[0]
# train_1[:,1] = (train_1[:,1] - train_mu[1])/train_sigma[1]

# test_sigma = np.std(test_1,axis=0,ddof=1)#标准差,axis=0计算每一列
# test_mu = np.mean(test_1,axis=0)#求平均值
# test_1[:,0] = (test_1[:,0] - test_mu[0])/test_sigma[0]
# test_1[:,1] = (test_1[:,1] - test_mu[1])/test_sigma[1]

# def svmTrain(train, C, kernal, r):
# 	X = train[:,]

# def svmTest(svm, test):
# 	n = len(test)

# def train(x):
# 	y = x[:,2]
# 	x = x[:,0:2]
# 	print(x,y)

