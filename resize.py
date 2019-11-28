import numpy as np
from scipy import ndimage, misc
import os, os.path
import sys
import matplotlib.pyplot as plt
import cv2
from skimage.color import rgb2gray

# print (cv2.__version__)


# This is a common file for all the other python scripts and is not to be run individually.
# For different other python scripts, 1-2 lines from this code should be commented/uncommented. 
# Those instructions are given under the run instructions of other files.



hog = cv2.HOGDescriptor()

def create_features(img):
    
    color_features = img.flatten()                                            # flatten three channel color image
    # hog_features = hog(img, block_norm='L2-Hys', pixels_per_cell=(16, 16))    # get HOG features from greyscale image  
    hog_features = hog.compute(img)
    flat_features = np.hstack(color_features)                                 # combine color and hog features into a single array
    
    return flat_features


def resizefunc(src, images, fin_im, output,label,xcoord,ycoord):
  onlyfiles = [ f for f in os.listdir(src) if os.path.isfile(os.path.join(src,f)) ]

  for n in range(0, len(onlyfiles)):
    img = cv2.imread(os.path.join(src,onlyfiles[n]))        #pick one image from dataset
    
    fin_im.append(onlyfiles[n])
    
    ###if you want gey images uncomment next 4 lines##
    # cv2.imshow(os.path.join(src,onlyfiles[n]), img)
    # cv2.waitKey(1000)
    # gray = rgb2gray(img)
    # plt.imshow(gray, cmap='gray')
    
    # cv2.imshow(os.path.join(src,onlyfiles[n]), gray)
    # cv2.waitKey(1000)

    im  = misc.imresize(img,(xcoord,ycoord))                #resize
    images.append(im)                                       #append an image as a whole

    #  flat_im = create_features(im)                           #flat image of 64*128image
    #  fin_im.append(flat_im)                                  #final list of all flat images

    output.append(label)
    print n

# len(onlyfiles)

# src = "/home/xyz/Desktop/XYZ/Dataset/Walls/Brickwork"
# images1 = []
# fin_im1 = []
# output = []             # labels --> Y matrix
# resizefunc(src=src, images = images1, fin_im=fin_im1, output=output, label=1, xcoord=128, ycoord=128)

# print "Plastering"
# src = "/home/xyz/Desktop/XYZ/Dataset/Walls/Plastering"
# resizefunc(src=src, images = images1, fin_im=fin_im1, output=output, label=1, xcoord=128, ycoord=128)

# print "over"
# cv2.waitKey(5000)
