import numpy as np
def SMO(x, y, C):
	# x = np.matrix(x)
	length = len(y)
	alpha = np.zeros(length)
	b = 0
	tol = 1e-7
	while 1:
		num_changed_alphas = 0
		xt = np.transpose(x)
		xt = np.matrix(xt)
		#计算内积
		zij = x * xt
		for i in range(length):
			Ei = alpha * y * zij[:,i] + b - y[i]
			#不满足kkt条件
			if (y[i] * Ei < -tol and alpha[i] < C) or (y[i] * Ei > tol and alpha[i] > 0):
				for j in range(0,i):
					Ej = alpha * y * zij[:,j] + b - y[j]
					alpha_old_i = alpha[i]
					alpha_old_j = alpha[j]
					if y[i] == y[j]:
						L = np.maximum(0, alpha[i] + alpha[j] - C)
						H = np.minimum(C, alpha[j] + alpha[i])
					else:
						L = np.maximum(0, alpha[j] - alpha[i])
						H = np.minimum(C, C + alpha[j] - alpha[i])
					if L == H:
						continue
					yita = zij[i,j] * 2 - zij[i,i] - zij[j,j]
					if yita >= 0:
						continue
					alpha[j] = alpha[j] - y[j] * (Ei - Ej) / yita
					if alpha[j] > H:
						alpha[j] = H
					elif alpha[j] < L:
						alpha[j] = L
					if np.linalg.norm(alpha[j] - alpha_old_j) < tol:
						continue
					alpha[i] = alpha[i] + y[i] * y[j] * (alpha_old_j - alpha[j])
					b1 = b - Ei - y[i] * (alpha[i] - alpha_old_i) * zij[i,i] - y[j] * (alpha[j]-alpha_old_j) * zij[i,j]
					b2 = b - Ej - y[i] * (alpha[i] - alpha_old_i) * zij[i,j] - y[j] * (alpha[j]-alpha_old_j) * zij[j,j]
					if alpha[i] < C and alpha[i] > 0:
						b = b1
					elif alpha[j] < C and alpha[j] > 0:
						b = b2
					else:
						b = (b1 + b2) / 2
					#重新计算Ei,如果还不满足kkt条件,继续,如果满足则寻找下一个i
					Ei = alpha * y * zij[:,i] + b - y[i]
					if (y[i] * Ei < -tol and alpha[i] < C) or (y[i] * Ei > tol and alpha[i] > 0):
						num_changed_alphas = num_changed_alphas + 1
					else:
						break
		#去除alpha==0和alpha==c
		x1 = []
		y1 = []
		alpha1 = []
		for k in range(len(alpha)):
			if alpha[k] != 0 and alpha[k] != C:
				# la[k] = 1
				x1.append(x[k])
				y1.append(y[k])
				alpha1.append(alpha[k])

		x = x1
		y = np.array(y1)
		alpha = np.array(alpha1)
		length = len(y)
		if num_changed_alphas == 0:
			break
	return alpha,x,y,b	#x:支持向量，y:label