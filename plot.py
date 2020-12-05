
import numpy as np
import matplotlib.pyplot as plt

from func import function

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
