import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np
import datetime

value = img.imread('D:\\GitHub\\Machine-Learning\\K-Means\\bird_small.tiff')
row = np.size(value,0)
col = np.size(value,1)
#u为随机采样16种颜色RGB值作为聚类
u = np.zeros((16,3))
for i in range(16):
	for j in range(3):
		u[i,j] = np.random.randint(255)
print(u)
o_length = np.zeros(16)
label = np.zeros((row,col))
oldu = np.zeros((16,3))
starttime = datetime.datetime.now()
for iter in range(100):
	et3 = datetime.datetime.now()
	for i in range(row):
		for j in range(col):
			for k in range(16):#计算每个点到 16 个初始聚点的欧式距离,找到最小距离分为一类
				o_length[k] = ((value[i,j,0]-u[k,0])**2+(value[i,j,1]-u[k,1])**2+(value[i,j,2]-u[k,2])**2)**0.5
			label[i,j] = o_length.tolist().index(min(o_length))
			# print(label)
	for i in range(16):
		for j in range(3):
			oldu[i,j] = u[i,j]#保存原聚点
	
	for k in range(16):
		count = 0#计算每个颜色有多少个
		total = np.zeros(3)
		for i in range(row):
			for j in range(col):
				if label[i,j] == k:
					total = total + [value[i,j,0],value[i,j,1],value[i,j,2]]
					count += 1
		if count != 0:
			u[k,:] = total/count#计算每一类的平均值，将平均值作为下一次迭代的聚点
			
	#退出条件，当u的差别不大时退出
	len_u = 0
	for i in range(16):
		for j in range(3):
			len_u = len_u + ((u[i,j] - oldu[i,j])**2)**0.5
	if len_u < 1e-6:
		break
	et4 = datetime.datetime.now()
	print("第%s次迭代,耗时：%s" %(iter,et4-et3))
et1 = datetime.datetime.now()
print("迭代总时间：%s" %(et1 - starttime))

lvalue = img.imread('D:\\GitHub\\Machine-Learning\\K-Means\\bird_large.tiff')
lvalue.flags.writeable = True#图像是只读模式,修改权限
lrow = np.size(lvalue,0)
lcol = np.size(lvalue,1)
lo_length = np.zeros(16)
for i in range(lrow):
	for j in range(lcol):
		for k in range(16):#计算原图每个像素到16个迭代之后的据点的距离，找到最小距离分为一类
			lo_length[k] = ((lvalue[i,j,0]-u[k,0])**2+(lvalue[i,j,1]-u[k,1])**2+(lvalue[i,j,2]-u[k,2])**2)**0.5
		lvalue[i,j,:] = u[lo_length.tolist().index(min(lo_length)),:]#用聚点值替换该类像素RGB值
et2 = datetime.datetime.now()
print("替换原图耗时：%s" %(et2 - et1))
print("总耗时：%s" %(et2 - starttime))
plt.imshow(lvalue)
plt.show()
plt.imsave('bird_kmeans.tiff',lvalue)