import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np
import datetime

value = img.imread('D:\\GitHub\\Machine-Learning\\K-Means\\bird_small.tiff')
row = np.size(value,0)
col = np.size(value,1)
#u为随机采样16种颜色作为聚类
u = np.zeros((16,3))
for i in range(16):
	for j in range(3):
		u[i,j] = np.random.randint(255)
print(u)
o_length = np.zeros(16)
label = np.zeros((row,col))

starttime = datetime.datetime.now()
for iter in range(100):
	for i in range(row):
		for j in range(col):
			for k in range(16):#计算每个点到 16 个初始聚点的欧式距离,然后找出最小的
				o_length[k] = ((value[i,j,0]-u[k,0])**2+(value[i,j,1]-u[k,1])**2+(value[i,j,2]-u[k,2])**2)**0.5
			label[i,j] = o_length.tolist().index(min(o_length))
			# print(label)
	old_u = u
	for k in range(16):
		count = 0#计算每个颜色有多少个
		total = np.zeros(3)
		for i in range(row):
			for j in range(col):
				if label[i,j] == k:
					total = total + [value[i,j,0],value[i,j,1],value[i,j,2]]
					count += 1
		if count == 0:
			continue
		else:
			u[k,:] = total/count
	#退出条件，当u的差别不大时退出
	len_u = 0
	for i in range(16):
		for j in range(3):
			len_u = len_u + ((u[i,j] - old_u[i,j])**2)**0.5
	if len_u < 1e-6:
		print(iter)
		break
et1 = datetime.datetime.now()
print(et1 - starttime)

lvalue = img.imread('D:\\GitHub\\Machine-Learning\\K-Means\\bird_large.tiff')
lvalue.flags.writeable = True#图像是只读模式,修改权限
lrow = np.size(lvalue,0)
lcol = np.size(lvalue,1)
lo_length = np.zeros(16)
for i in range(lrow):
	for j in range(lcol):
		for k in range(16):
			lo_length[k] = ((lvalue[i,j,0]-u[k,0])**2+(lvalue[i,j,1]-u[k,1])**2+(lvalue[i,j,2]-u[k,2])**2)**0.5
		lvalue[i,j,:] = u[lo_length.tolist().index(min(lo_length)),:]
et2 = datetime.datetime.now()
print(et2 - et1)
plt.imshow(lvalue)
plt.show()