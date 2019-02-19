import numpy as np

def naive_vector_dot(x,y):
	assert len(x.shape) == 1
	assert len(y.shape) == 1
	assert x.shape[0] == y.shape [0]
	z = 0.
	for i in range(x.shape[0]):
		z += x[i] * y[i]
	return z

def naive_matrix_vector_dot1(x,y):
	assert len(x.shape) == 2
	assert len(y.shape) == 1
	assert x.shape[1] == y.shape[0]
	z = np.zeros(x.shape[0])
	for i in range(x.shape[0]):
		for j in range(x.shape[1]):
			z[i] += x[i,j] * y[j]
	return z

def naive_matrix_vector_dot2(x,y):
	z = np.zeros(x.shape[0])
	for i in range(x.shape[0]):
		z[i] = naive_vector_dot(x[i,:],y)
	return z

def naive_matrix_dot(x,y):
	assert len(x.shape) == 2
	assert len(y.shape) == 2
	assert x.shape[1] == y.shape[0]
	z = np.zeros((x.shape[0],y.shape[1]))
	for i in range(x.shape[0]):
		for j in range(y.shape[1]):
			row_x = x[i,:]
			column_y = y[:,j]
			z[i,j] = naive_vector_dot(row_x,column_y)
	return z

x = np.random.random((10,20))
y = np.random.random((20,))
y1 = np.random.random((20,30))
print(naive_matrix_vector_dot1(x,y))
print('='*30)
print(naive_matrix_vector_dot2(x,y))
print(naive_matrix_dot(x,y1))
print('='*30)
print(x.dot(y1))


# 更高维张量做点积,只要其形状匹配遵循与前面2D张量相同原则
# (a,b,c,d) . (d,)  ->  (a,b,c)
# (a,b,c,d) . (d,e) ->  (a,b,c,e)
