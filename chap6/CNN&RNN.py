import os 
import numpy as np
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #忽略报错
# 打开数据
name = 'jena_climate_2009_2016.csv'
with open(name) as fp:
	data = fp.read()
lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]
# print(header)
# print(len(lines))

# 解析数据
float_data = np.zeros((len(lines),len(header) - 1))
for i,line in enumerate(lines):
	values = [float(x) for x in line.split(',')[1:]]
	float_data[i,:] = values

# temp = float_data[:,1]
# plt.figure(1)
# plt.plot(range(len(temp)),temp)
# plt.figure(2)
# plt.plot(range(1440),temp[:1440])
# plt.show()

# 数据标准化
mean = float_data[:200000].mean(axis=0)
float_data -= mean
std = float_data[:200000].std(axis=0)
float_data /= std

#生成时间序列样本及目标的生成器
def generator(data,lookback,delay,min_index,max_index,shuffle=False,batch_size=128,step=6):
	if max_index is None:
		max_index = len(data) - delay -1
	i = min_index + lookback
	while 1:
		if shuffle:
			rows = np.random.randint(min_index + lookback,max_index,size=batch_size)
		else:
			if i + batch_size >= max_index:
				i = min_index + lookback
			rows = np.arange(i,min(i + batch_size,max_index))
			i += len(rows)
		samples = np.zeros((len(rows),lookback//step,data.shape[-1]))
		targets = np.zeros((len(rows),))
		for j,row in enumerate(rows):
			indices = range(rows[j] - lookback,rows[j],step)
			samples[j] = data[indices]
			targets[j] = data[rows[j] + delay][1]
		yield samples,targets

#准备更高分辨率的数据生成器
step = 3
lookback = 720
delay = 144
train_gen = generator(float_data,lookback=lookback,delay=delay,min_index=0,max_index=200000,shuffle=True,step=step)
val_gen = generator(float_data,lookback=lookback,delay=delay,min_index=200001,max_index=300000,step=step)
test_gen = generator(float_data,lookback=lookback,delay=delay,min_index=300001,max_index=None,step=step)
val_steps = (300000 - 200001 - lookback) // 128
test_steps = (len(float_data) - 300001 - lookback) // 128

#构建CNN&RNN
model = Sequential()
model.add(layers.Conv1D(32,5,activation='relu',input_shape=(None,float_data.shape[-1])))
model.add(layers.MaxPooling1D(3))
model.add(layers.Conv1D(32,5,activation='relu'))
model.add(layers.GRU(32,dropout=0.1,recurrent_dropout=0.5))
model.add(layers.Dense(1))
# print(model.summary())
model.compile(optimizer=RMSprop(),loss='mae')
history = model.fit_generator(train_gen,steps_per_epoch=500,epochs=20,validation_data=val_gen,validation_steps=val_steps)

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1,len(loss_values)+1)
plt.figure(1)
plt.plot(epochs,loss_values,'bo',label="Training loss")
plt.plot(epochs,val_loss_values,'b',label="validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()