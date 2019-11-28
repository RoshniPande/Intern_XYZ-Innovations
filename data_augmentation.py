import random
from scipy import ndarray
import skimage as sk
from skimage import transform
from skimage import util
import cv2
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage import img_as_float


				##### RENAME ALL PICS IF NECESSARY. USE COMMAND-->  rename 's/abc/xyz/' *    FOR abc-->xyz ######
													##### CHANGE SAVE PREFIX ######

src = "/home/xyz/Desktop/XYZ/Dataset/Walls/Plastering/new"
num_samples = 10

onlyfiles = [ f for f in os.listdir(src) if os.path.isfile(os.path.join(src,f)) ]

datagen = ImageDataGenerator(
		rescale= 1./255,
        width_shift_range=0.3,
        height_shift_range=0.3,        
        shear_range=0.2,
        zoom_range=0.25,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.5, 1.05],
        fill_mode='wrap')

# print("length: ")
# print len(onlyfiles)

for k in range(len(onlyfiles)):
	im = cv2.imread(os.path.join(src,onlyfiles[k])) 
	img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
	# img= im.astype(np.uint8) 
	x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
	x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
	print x.shape
	# print img.dtype

	i = 0
	# for batch in datagen.flow(x, batch_size=1, save_to_dir='Brickwork', save_prefix='aug_', save_format='jpeg'):
	for batch in datagen.flow(x, batch_size=1, save_to_dir=src, save_prefix='p_aug', save_format='jpeg'):
	    i += 1
	    if i > num_samples:
	        break  # otherwise the generator would loop indefinitely