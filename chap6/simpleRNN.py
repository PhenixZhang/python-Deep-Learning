import numpy as np

timesteps = 100
input_features = 32
output_features = 64

inputs = np.random.random((timesteps,input_features))
state_t = np.zeros((output_features,))
W = np.random.random((output_features,input_features)) # input
U = np.random.random((output_features,output_features)) #state
b = np.random.random((output_features,)) # bias

successive_outputs = []
for input_t in inputs:
	output_t = np.tanh(np.dot(W,input_t) + np.dot(U,state_t) + b)
	successive_outputs.append(output_t)
	state_t = output_t
final_output_sequence = np.stack(successive_outputs,axis=0)