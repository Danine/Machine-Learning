import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np

value = img.imread('D:\\GitHub\\Machine-Learning\\K-Means\\bird_small.tiff')
row = np.size(value,0)
col = np.size(value,1)
print(value.shape)
#u为随机采样16种颜色作为聚类
u = np.zeros((16,3))
for i in range(16):
	for j in range(3):
		u[i,j] = np.random.randint(255)
print(u)
for iter in range(100):
	for i in range(row):
		for j in range(col):
			a = 0
# plt.imshow(value)
# plt.show()