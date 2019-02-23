import numpy as np
def naive_add_matrix_and_vector(x,y):
	assert len(x.shape) == 2
	assert len(y.shape) == 1
	assert x.shape[1] == y.shape[0]

	x = x.copy()
	for i in range(x.shape[0]):
		for j in range(x.shape[1]):
			x[i,j] += y[j]
	return x

x = np.random.random((32,10))
y = np.random.random((10,))
# print(x)
# print("="*100)
# print(y)
# print("="*100)
# print(len(x.shape),len(y.shape))
print(naive_add_matrix_and_vector(x,y))