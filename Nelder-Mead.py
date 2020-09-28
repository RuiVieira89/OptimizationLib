
## Rui Vieira, 2020
## Optimization without derivatives

'''Nelder-Mead Method'''

# iterative method that define a simplex(S_k) - Polyhedra
# for a dimesion R**n it needs n+1 points X(1) -> X(n+1)
# S_k = [X(1), ... , X(n+1)] such as they are ordered
# that is f(X(1)) <= f(X(2)) ... 
# 1 is best, n+1 is the worst

## Matrix A is organised as:
# A [[X1,     fX1]
#    [X2,     fX2]
#    [Xn+1, fXn+1]]

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def function(x):
	return np.maximum((x[0] - 1)**2, x[0]**2 + 4*(x[1]-1)**2)


def plotSurf(A, minimum):


	delta = 0.1
	x = y = np.arange(-10.0, 10.0, delta)
	X,Y = np.meshgrid(x, y)
	Z = function([X, Y])

	plt.ion()
	fig = plt.figure(1, figsize=(10,8), clear=True)
	ax = fig.add_subplot(1, 1, 1)


	im = ax.contourf(X, Y, Z, levels=40)
	ax.scatter(A[:,0], A[:,1], marker='x', c='r')

	for i in range(0,A.shape[0]):
		s = f"X{i}"
		ax.text(A[i,0], A[i,1], s, fontsize=12)

	s1 = 'Min'
	ax.scatter(minimum.x[0], minimum.x[1], marker='o', c='r')
	ax.text(minimum.x[0], minimum.x[1], s1, fontsize=12)


	fig.colorbar(im,ax=ax)

	plt.pause(0.1)


def NelderMead():
	minimum = minimize(function, [1,1]) # for comparison

	n = 2 # dimensions
	A = np.random.uniform(10, -10, (n+1, n+1))

	err = 1
	tol = 1e-8
	while err > tol:

		for i in range(0, A.shape[0]):
			A[i,-1] = function(A[i,0:-1])

		A = A[A[:,-1].argsort()]

		# centroid calculation
		X_ave = np.average(A[0:-1,0:-1], axis=0)

		# Refleted vertex (alfa=1)
		alfa = 1
		X_r = (1 + alfa)*X_ave - alfa*A[-1,0:-1]


		# Expanded vertex (gamma=2)
		gamma = 2
		Xe = gamma*X_r + (1 - gamma)*X_ave

		# contracred vertex (beta=0.5)
		beta=0.5
		Xc = beta*X_r + (1 - beta)*X_ave

		# interior contracred vertex (beta=0.5)
		beta=0.5
		Xc = beta*A[-1,0:-1] + (1 - beta)*X_ave

		print('')

		token = 0
		# Very Good
		if (token == 0) and (function(X_r) < A[0,-1]):
			if function(Xe) < A[0,-1]:
				#accept Xe
				A[-1,0:-1] = Xe
				token = 1
				print('Expanded vertex')
			else:
				#accept X_r
				A[-1,0:-1] = X_r
				token = 1
				print('Refleted vertex')

		# Good
		if (token == 0) and (function(A[0,0:-1]) <= function(X_r)) and (function(X_r) < function(A[-2,0:-1])):
			#accept X_r
			A[-1,0:-1] = X_r
			token = 1
			print('Good')

		# Bad
		if (token == 0) and (function(A[-2,0:-1]) <= function(X_r)) and (function(X_r) < function(A[-1,0:-1])):
			if function(Xc) < function(A[-2,0:-1]):
				#accept Xc
				A[-1,0:-1] = Xc
				token = 1
				print('interior contracred vertex')
			else:
				#choose S(k+1)
				# Choose Simplex
				for i in range(2, A.shape[0]):
					A[i,0:-1] = (A[i,0:-1] + A[0,0:-1])/2
				token = 1
				print('Simplex')

		#Very Bad
		if (token == 0) and function(X_r) >= function(A[-1,0:-1]):
			if function(Xc) < function(A[-2,0:-1]):
				#accept Xc
				A[-1,0:-1] = Xc
				token = 1
				print('interior contracred vertex')
			else:
				#choose S(k+1)
				# Choose Simplex
				for i in range(2, A.shape[0]):
					A[i,0:-1] = (A[i,0:-1] + A[0,0:-1])/2
				token = 1
				print('Simplex')


		# stop criterium
		delta = max(1, np.linalg.norm(A[0,0:-1]))
		e = np.zeros([n+1])
		for i in range(1,A.shape[0]):
			e[i-1] = np.linalg.norm(A[i,0:-1] - A[0,0:-1])
		err = (1/delta)*max(e)
		
		print(err)
		plotSurf(A, minimum)


	print('\nOpt\n',f'X={minimum.x}; fX={function(minimum.x)}\n')
	print('NelderMead\n',f'X={A[0,0:-1]}; fX={function(A[0,0:-1])}; relSize={err}')



def main():

	NelderMead()


if __name__ == '__main__':

	main()

