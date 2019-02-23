import os 
from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #忽略报错
(train_images,train_labels),(test_images,test_labels) = mnist.load_data() #导入训练集和测试集
# print(train_images.shape)
# digit = train_images[4]
# plt.imshow(digit,cmap=plt.cm.binary) #通过图像显示数字图片
# plt.show()

#构建神经网络
network = models.Sequential() #初始化
network.add(layers.Dense(512,activation='relu',input_shape=(28*28,))) #添加隐藏层
network.add(layers.Dense(10,activation='softmax')) #添加输出层
network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy']) #选择优化器、损失函数、评估标准

#向量化数据
train_images = train_images.reshape((60000,28*28))
train_images = train_images.astype('float32')/255
test_images = test_images.reshape((10000,28*28))
test_images = test_images.astype('float32')/255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

network.fit(train_images,train_labels,epochs=5,batch_size=128) #拟合数据
test_loss,test_acc = network.evaluate(test_images,test_labels) #评估预测标准

#输出结果
print('test_loss:',test_loss)
print('test_acc:',test_acc)