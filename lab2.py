import numpy as np
from numpy.linalg import pinv
from PIL import Image
import matplotlib.pyplot as plt

def norm(A):
	return np.sqrt((A.getT() * A).trace())

def SVD(A):
	U, D, V = np.linalg.svd(A, full_matrices = True)
	return U, D, V

def SVD_inv(A):
	U, d, V = SVD(A)
	m = len(U)
	n = len(V)

	D = np.array([[0.0 for i in range(n)] for j in range(m)])
	D_inv = np.array([[0.0 for i in range(m)] for j in range(n)])
	for i in range(len(d)):
		if np.abs(d[i]) > 1e-6:
			D[i][i] = d[i]
			D_inv[i][i] = 1.0 / d[i]
		else:
			D[i][i] = 0.0
			D_inv[i][i] = 0.0

	return V.getT() * D_inv * U.getT()

def greville(A, eps = 0):
	p_inv = np.array([])
	m = len(A)
	a = A[:,0]



def pseudo_inv(A, eps = 0): #A is np matrix
	I = np.identity(len(A.getT()))

	A_pinv = (A.getT() * A + (eps * I)).getI() * A.getT()

	return A_pinv

y1_bmp = Image.open('y3.bmp')
y2_bmp = Image.open('y4.bmp')

print(y1_bmp)



X = np.array(y1_bmp)

#plt.imsave('Y3.png', X)


Y = np.array(y2_bmp)

print(X)

#img = Image.fromarray(Y1, 'RGB')

#A = Y * pseudo_inv(np.matrix(X))

MX = np.matrix(X)
MY = np.matrix(Y)

#print(MX.getT() * MX)

X_inv = pseudo_inv(MX, eps = 1)

#X_inv = np.linalg.pinv(MX, rcond = 1e-15)

X_inv_svd = SVD_inv(MX)


print(X_inv)

A = MY * X_inv
A1 = MY * X_inv_svd


Y1 = A * MX
Y2 = A1 * MX

plt.imsave('Y1.png', Y1, cmap = plt.cm.gray)
plt.imsave('Y2.png', Y2, cmap = plt.cm.gray)
