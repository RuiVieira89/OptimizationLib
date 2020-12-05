
import numpy as np

def function(x):
	return np.maximum((x[0] - 1)**2, x[0]**2 + 4*(x[1]-1)**2)
