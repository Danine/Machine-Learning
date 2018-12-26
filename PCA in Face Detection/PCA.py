import numpy as np
from scipy.sparse import linalg
import matplotlib.image as img
import random

def loadImageSet():
	trainData = []; testData = []; trainLabel = []; testLabel = [] 
	path = 'D:\\GitHub\\Machine-Learning\\PCA in Face Detection\\orl_faces'
	for i in range(40):
		count = random.sample(range(10),6)#无重复随机6个0~9的数
		data = [img.imread(path + '\\s%d\\%d.pgm' %(i+1,j+1)) for j in range(10)]#每次循环读入一个文件夹里的10张人脸
		trainData.extend(data[j].ravel() for j in count)#随机的6张作为训练集
		testData.extend(data[j].ravel() for j in range(10) if j not in count)#剩下的4张作为测试集
		trainLabel.extend([i] * 6)#读入6个当前i作为训练集label
		testLabel.extend([i] * 4)#读入4个当前i作为测试集label
	return np.array(trainData), np.array(testData), np.array(trainLabel), np.array(testLabel)

def pca(data,k):
	m = np.size(data)
	trainMean = np.mean(data,0)
	data = data - trainMean#数据归一化，即减去平均值
	data = np.mat(data)#转为矩阵
	S = data.T * data / m#计算协方差矩阵
	U = linalg.eigs(S,k)#计算特征值
	Z = data * U[1]#得到特征矩阵
	return np.array(Z),trainMean,U

def main():
	trainD, testD, trainL, testL = loadImageSet()#读入图片集
	k = 50#取特征的数量，即降维
	Z,trainMean,U = pca(trainD,k)
	testD = testD - trainMean#对测试集进行归一化
	testD = np.mat(testD)
	Z1 = np.array(testD * U[1])#得到测试集的特征矩阵

	testP = [trainL[np.sum((Z-d)**2,1).argmin()]for d in Z1]#欧式距离法得到测试集的预测值label
	print((testP == testL).mean())

	# testP = []
	# o_length = np.zeros(240)
	# for d in range(len(Z1)):
	# 	for j in range(240):
	# 		o_length[j] = (np.sum((Z[j]-Z1[d])**2))**0.5
	# 	b = o_length.tolist().index(min(o_length))
	# 	a = trainL[b]
	# 	testP.append(a)
	# print((testP == testL).mean())

if __name__ == '__main__':
	main()