import numpy as np
def naive_relu(x):
	assert len(x.shape) == 2
	x = x.copy()
	for i in range(x.shape[0]):
		for j in range(x.shape[1]):
			x[i,j] = max(x[i,j],0)
	return x

def naive_add(x,y):
	assert len(x.shape) == 2
	assert x.shape == y.shape
	x = x.copy()
	for i in range(x.shape[0]):
		for j in range(x.shape[1]):
			x[i,j] += y[i,j]
	return x

x = np.array([[1,-0.4,0.5],[0.3,-0.6,0.8],[0.7,-0.4,0.9]])

print(naive_relu(x))
print(np.maximum(x,0))

print(naive_add(x,x))
print(x + x)

