import cv2 
import numpy as np 
from keras import backend as K
from keras.models import load_model
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

import resize

################# load model##################
# model = load_model('model.h5')
model = load_model('model.h5')

# summarize model.
model.summary()

############## load dataset #################
output = []				# labels --> Y matrix
images = []
flat_im_list = []
fin_im_list = []
xdim = 128
ydim = 128

				##### CHANGE RESIZE FUNCTION FIN_IM. UNCOMMENT THAT LINE IF COMMENTED######
src = "/home/xyz/Desktop/XYZ/Dataset/Walls/test/bricks"
resize.resizefunc(src=src, images = images, fin_im=fin_im_list, output=output, label=0, xcoord=xdim, ycoord=ydim)
print(len(images), len(fin_im_list), len(output))

src = "/home/xyz/Desktop/XYZ/Dataset/Walls/test/plaster"
resize.resizefunc(src=src, images = images, fin_im=fin_im_list, output=output, label=1, xcoord=xdim, ycoord=ydim)
print(len(images), len(fin_im_list), len(output))

k=len(images)
print ("no. of images: ",k)
X=np.zeros((k,xdim,ydim,3))

for i in range(k):
	X[i,:,:,:]=images[i]

X = X.astype('float32') / 255

print (X.shape)
y = np.array(output)
# X, y = shuffle(X, y)

########## evaluate model #################

y_pred=model.predict(X)

for i in range(0,len(y_pred)):
	if(y_pred[i] < 0.5): 
		y_pred[i] = 0
	else:
		y_pred[i] = 1

# print y_pred
accuracy = accuracy_score(y, y_pred)				# calculate accuracy
# print (y.shape, len(fin_im_list), len(y_pred))

for i in range(0,k):
	print (fin_im_list[i]," actual:", y[i], " pred:",y_pred[i])

score = model.evaluate(X, y, verbose=0)
print("%s: %.5f%%" % (model.metrics_names[1], score[1]*100))