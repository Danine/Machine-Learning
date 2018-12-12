import numpy as np
train_data = np.loadtxt('D:/GitHub/Machine-Learning/NaiveBayes/training_data.txt',dtype=int)
test_data = np.loadtxt('D:/GitHub/Machine-Learning/NaiveBayes/test_data.txt',dtype=int)

Py = []
# Bayes = np.zeros((5,8,5), dtype = float)
LBayes = np.zeros((5,8,5), dtype = float)

def train(train_data):
	Total = len(train_data[:,8])
	Class = train_data[:,8]
	total_class = []
	for i in range(5):	#for 5 class
		total_class.append(np.sum(Class == i))	#sum of each class
		Py.append(total_class[i]/Total)	#P(each class)
		lam = 1#for Laplace Smoothing 
		for j in range(8):	#for 8 features per item
			for k in range(5):	#for 5 situations per feature
				xlength = np.sum((train_data[:,j] == k) & (Class == i))
				# Pijk = xlength/total_class[i]
				LPijk = (xlength + lam)/(total_class[i] + 5*lam)
				# Bayes[i,j,k] = Pijk
				LBayes[i,j,k] = LPijk

def test(test_data,total):
	Total_test = len(test_data[:,8])
	prediction = np.zeros((Total_test,1),dtype = int)
	for i in range(Total_test):
		preValue = 0
		cmp1 = 0
		for j in range(5):
			refer = 1
			for k in range(8):
				refer = refer * LBayes[j,k,(test_data[i,k])]
			cmp2 = Py[j] * refer
			if cmp2 > cmp1:	#find the max of 5 classes
				preValue = j
				cmp1 = cmp2
		prediction[i,0] = preValue
	key = 0
	for i in range(Total_test):
		if prediction[i,0] == test_data[i,8]:
			key += 1
	print("%s\t\t%s" %(total,key/Total_test))

print("the first question")
train(train_data)
test(test_data,10000)
print("the second question")
Py = []
x2 = train_data[np.random.randint(10000, size=10),:]
train(x2)
test(test_data,10)

Py = []
x3 = train_data[np.random.randint(10000, size=100),:]
train(x3)
test(test_data,100)

Py = []
x4 = train_data[np.random.randint(10000, size=1000),:]
train(x4)
test(test_data,1000)

Py = []
x5 = train_data[np.random.randint(10000, size=1000),:]
train(x5)
test(train_data[np.random.randint(10000, size=1000),:],1000)