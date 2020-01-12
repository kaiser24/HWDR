from keras.datasets import mnist
import cv2
import numpy as np
import matplotlib.pyplot as plt

SHOWSAMPLE = False


(x_train, y_train),(x_test, y_test) = mnist.load_data()  #loading the mnist database of hand written digits


if(SHOWSAMPLE):
	#displaying a random digit
	random_digit = np.random.randint(0, x_train.shape[0])
	cv2.imshow('Random', x_train[random_digit])
	cv2.waitKey(0)
	cv2.destroyAllWindows()

#for some reason keras needs the data to be in the shape ndata, imrows, imcols, depth
#where depth is the number of layers of the image. 1 if its a grayscale image, 3 if its colored.

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

input_shape = (x_train[0].shape[0], x_train.shape[1], 1)

#now we have to normilize our images

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255


#our label arrays need to be in hot encoded form. keras
#provides a function to transform our information to hot encoded

#hot encoded is: lets say we have 3 classes, from 0 to 2, and our label array
#has  6 items. each item could be 0, 1 or 2. in hot one endoding that information
#becomes an array 6,3 where each row has only the label item as 1 and the rest as 0
# example     Array    Hot one   
#               1       0 1 0
#               2       0 0 1
#               0       1 0 0
#               1       0 1 0

from keras.utils import np_utils

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

Nclasses = y_train.shape[1]  #number of classes of this problem.

#now we create our CNN model
import keras
from keras.models import Sequential  #to add layers to our CNN as desired
from keras.layers import Dense, Dropout, Flatten #utils layers.
#Dense is a fully connected layer, flatten takes the matrix of the previous stage and flattens it to pass it 
#to a fully connected layer. and droput is to reduce overfitting(?)
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D  #Convolutional layers
from keras.optimizers import SGD

#creating our model
model = Sequential()

#our model will be as follows:

# Convolutional with 32 filters => Convolutional with 64 filters =>Max Pooling => Dropout 
#=> Flatten => FCL 128 neurons => Dropout => FCL number of classes 


#adding layers to our model
model.add(Conv2D(32, kernel_size=(3,3),	activation='relu', input_shape=input_shape))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(Nclasses, activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = SGD(0.01), metrics = ['accuracy'])

summary = model.summary()

print(summary)

#====================================Trainning=============================
batch_size = 64
epochs = 20

print('=========TRAINNING=======')
history = model.fit(
	x_train,
	y_train,
	batch_size=batch_size,
	epochs=epochs,
	verbose=1,
	validation_data=(x_test,y_test)
)
print('Trainning Done')

print('Testing')
score = model.evaluate(x_test, y_test, verbose=0)
print(score)

history_dict = history.history
tr_loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range( 1, len( val_loss ) + 1 )
model.save('/home/felipe/nD/U de A/Machine Learning/Deep Learning/simple_cnn_20epochs.h5')

plt.plot(epochs, tr_loss , label = 'Training loss' )
plt.plot(epochs, val_loss, label =  'Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

