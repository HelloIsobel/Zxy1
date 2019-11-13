from keras import models
from keras import layers
from keras.datasets import mnist
from keras.utils import to_categorical

(train_data,train_labels),(test_data,test_labels)=mnist.load_data()

train_data=train_data.reshape((60000,28,28,1))
train_data=train_data.astype('float32')/255

test_data=test_data.reshape((10000,28,28,1))
test_data=test_data.astype('float32')/255

#将类别向量转换为二进制（只有0和1）的矩阵类型表示
train_labels=to_categorical(train_labels)
test_labels=to_categorical(test_labels)

model=models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64,(3,3),activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))

model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

#作图看是否过拟合，判断第几轮最优
model.fit(train_data,train_labels,epochs=5,batch_size=64)

train_score = model.evaluate(train_data,train_labels)
print('Accuracy of Training Set:', train_score[1])

test_score = model.evaluate(test_data,test_labels)
print('Accuracy of Testing Set:', test_score[1])