import numpy as np
from math import pi
import matplotlib.pyplot as plt

def fourier(y):
	N = len(y)
	sums = []

	for m in range(N):
		res = 0
		for k in range(N):
			res = res + y[k] * np.exp((-1j * 2 * pi * k * m) / N)
		sums.append(res / N)

	return sums

def function(a, t, f): #a is a np 5x1 matrix
	a1 = a.item(0, 0)
	a2 = a.item(1, 0)
	a3 = a.item(2, 0)
	a4 = a.item(3, 0)
	a5 = a.item(4, 0)
	return a1 * np.power(t, 3) + a2 * np.power(t, 2) + a3 * t + a4 * np.sin(2 * pi * f * t) + a5

def sum_t_deg(t, n):
	return sum([(np.power(x, n)) for x in t])

def sum_t_sin_deg(t, n, f):
	return sum([(np.power(x, n) * np.sin(2 * pi * f * x)) for x in t])

def sum_sin_squared(t, f):
	return sum([(np.power(np.sin(2 * pi * f * x), 2)) for x in t])

def sum_y_t_deg(y, t, n):
	return sum([a * np.power(b, n) for a, b in zip(y, t)])

def sum_y_sin(y, t, f):
	return sum([a * np.sin(2 * pi * f * b) for a, b in zip(y, t)])


f = open('f4.txt', 'r')
lines = f.readlines()
f.close()

y_samples = [float(number) for number in lines[0].split()]

C = fourier(y_samples)

t = [i / 100 for i in range(501)]
T = 5

plt.title('input')
plt.grid(True)
plt.ylim(-30, 230)
plt.xlim(0, 5)
plt.plot(t, y_samples)
plt.show()

plt.title('modules')
plt.grid(True)
plt.ylim(-2, 12)
plt.xlim(0, 5)
plt.plot(t, [abs(c) for c in C])
plt.show()

m_star = max([abs(c) for c in C[20:481]]) #function is not 
delta_f = 1 / T

print(m_star)

f1 = delta_f * m_star

t6 = sum_t_deg(t, 6)
t5 = sum_t_deg(t, 5)
t4 = sum_t_deg(t, 4)
t3 = sum_t_deg(t, 3)
t2 = sum_t_deg(t, 2)
t1 = sum_t_deg(t, 1)

t3sin = sum_t_sin_deg(t, 3, f1)
t2sin = sum_t_sin_deg(t, 2, f1)
t1sin = sum_t_sin_deg(t, 1, f1)
sin = sum_t_sin_deg(t, 0, f1)
sin_squared = sum_sin_squared(t, f1)

yt3 = sum_y_t_deg(y_samples, t, 3)
yt2 = sum_y_t_deg(y_samples, t, 2)
yt1 = sum_y_t_deg(y_samples, t, 1)
ysin = sum_y_sin(y_samples, t, f1)
yt0 = sum_y_t_deg(y_samples, t, 0)

B = np.matrix([
	[t6, t5, t4, t3sin, t3], 
	[t5, t4, t3, t2sin, t2], 
	[t4, t3, t2, t1sin, t1], 
	[t3sin, t2sin, t1sin, sin_squared, sin], 
	[t3, t2, t1, t1sin, 1]])

c = np.matrix([yt3, yt2, yt1, ysin, yt0])

a = B.getI() * c.getT()

func = [function(a, x, f1) for x in t]

plt.title('function')
plt.grid(True)
plt.plot(t, func)
plt.show()