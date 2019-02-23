import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
# import numpy as np
# x = np.zeros((300,20))
# print(x.T.shape)

from keras import models
from keras import layers

model = models.Sequential()


model.add(layers.Dense(32,input_shape=(784,)))
model.add(layers.Dense(32))