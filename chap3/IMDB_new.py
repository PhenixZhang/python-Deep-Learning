import numpy as np
from keras.datasets import imdb
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics
import matplotlib.pyplot as plt

def vectorize_sequences(sequences,dimension=10000):
	results = np.zeros((len(sequences),dimension))
	for i,sequence in enumerate(sequences):
		results[i,sequence] = 1.
	return results

(train_data,train_labels),(test_data,test_labels) = imdb.load_data(num_words=10000)
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

word_index = imdb.get_word_index()
reverse_word_index = dict([(value,key) for (key,value) in word_index.items()])
decoded_review = ' '.join([reverse_word_index.get(i -3,'?') for i in train_data[0]])
model = models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))
model.compile(optimizer=optimizers.RMSprop(lr=0.001),loss=losses.binary_crossentropy,metrics=[metrics.binary_accuracy])
model.fit(x_train,y_train,epochs=5,batch_size=512)
results = model.evaluate(x_test,y_test)
print(results)