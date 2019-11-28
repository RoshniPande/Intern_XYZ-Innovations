import cv2 
import numpy as np 
from random import shuffle 
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from keras.callbacks import ReduceLROnPlateau,EarlyStopping, ModelCheckpoint
from sklearn.utils import shuffle
from scipy import ndimage, misc
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Convolution2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as K

import resize

BATCH = 50
EPOCHS = 50

########## Data Preprocessing #########
output = []				# labels --> Y matrix
images = []
flat_im_list = []
xdim = 128
ydim = 128

				##### CHANGE RESIZE FUNCTION FIN_IM. COMMENT THAT LINE IF UNCOMMENTED######
src = "/home/xyz/Desktop/XYZ/Dataset/Walls/Brickwork"
resize.resizefunc(src=src, images = images, fin_im=flat_im_list, output=output, label=0, xcoord=xdim, ycoord=ydim)

src = "/home/xyz/Desktop/XYZ/Dataset/Walls/Plastering"
resize.resizefunc(src=src, images = images, fin_im=flat_im_list, output=output, label=1, xcoord=xdim, ycoord=ydim)

k=len(images)
print ("no. of images: ",k)
X=np.zeros((k,xdim,ydim,3))

for i in range(k):
	X[i,:,:,:]=images[i]

X = X.astype('float32') / 255

print X.shape
y = np.array(output)
print y.shape

X, y = shuffle(X, y, random_state=42)

########## Model training #########

x_train, x_test, y_train, y_test = train_test_split(X, 
													y, 
													random_state=42,
													test_size=.1, 
													)


model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=(xdim,ydim,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


model.add(Conv2D(24, (2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(32))

model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# opt = SGD(lr=0.08, momentum=0.9)
# model.compile(loss='binary_crossentropy',
#               optimizer='opt',
#               metrics=['accuracy'])

model.compile(loss='binary_crossentropy',
              optimizer='Adadelta',
              metrics=['accuracy'])

rlrop = ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.1, patience=7, verbose=1)
es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)
check = ModelCheckpoint(monitor='val_loss', mode='min', filepath='cnn_model.hdf5', save_best_only=True)
# model.fig(..., callbacks=[rlrop])

model.fit(x_train, y_train	
          ,batch_size=BATCH
          ,epochs=EPOCHS          
          ,verbose=1
          ,shuffle=True
          ,validation_split=0.15
          ,callbacks=[rlrop,es,check]
          )

y_pred=model.predict(x_test)
for i in range(0,len(y_pred)):
	if(y_pred[i] < 0.5): 
		y_pred[i] = 0
	else:
		y_pred[i] = 1
	
accuracy = accuracy_score(y_test, y_pred)				# calculate accuracy

print ('Model accuracy is: ')
print accuracy

model.summary()
model.save("model.h5")
# print X_train.ndimages
# time_start = time.time()