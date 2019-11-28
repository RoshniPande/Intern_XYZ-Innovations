Problem statement: Given an image of a wall under construction, we have to predict it's stage i.e, Brickwork stage or plastering stage.

Dataset has been collected from google for the two classes and has been augmented.

**resize.py** <br />
[resize.py](https://github.com/TonduriGopinath7/construction-stage-detection/blob/master/Walls/resize.py) is a common file for all the other python scripts in the repository and is not to be run individually. <br />
For different other python scripts, 1-3 lines from this code should be commented/uncommented. <br />
Those instructions are given under the run instructions of other files. <br />

**Walls_cnn.py** <br />
[This](https://github.com/TonduriGopinath7/construction-stage-detection/blob/master/Walls/Walls_cnn.py) is the code to run a convolutional neural network on the dataset [*Brickwork*](https://github.com/TonduriGopinath7/construction-stage-detection/blob/master/Walls/Brickwork.tar.gz) and [*Plastering*](https://github.com/TonduriGopinath7/construction-stage-detection/blob/master/Walls/Plastering.tar.gz). <br />
This code doesn't require the image to be converted to an array of flat features of (x_shape*y_shape, channels).
So comment all the 3 lines that have " *fin_im* ". <br />
- Change the src = "the source filepath to the dataset Brickwork/Plastering" <br />
- Comment lines 28, 40, 41 of *resize.py*

**user_cnn_walls.py** <br />
[This](https://github.com/TonduriGopinath7/construction-stage-detection/blob/master/Walls/user_cnn_walls.py) is the code for the user to feed the convolutional neural network on new dataset [*test*](https://github.com/TonduriGopinath7/construction-stage-detection/blob/master/Walls/test.tar.gz). <br />
The names of the misclassified images need to be seen, so uncomment fin_im that has appended the image names. <br />
- Change the src = "the source filepath to the test dataset" <br />
- comment lines 40, 41 of resize.py as flat features aren't required. <br />
- uncomment line 28 of resize.py as file names are required. <br />

**data_augmentation.py** <br />
[This](https://github.com/TonduriGopinath7/construction-stage-detection/blob/master/Walls/data_augmentation.py) code is used for data augmentation.<br />
You need to change the source of the images filepath. Also change the prefix with which you want the new images to be saved.<br /> For simplicity, brckwork images are prefixed with "aug" and plastering images are prefixed with "p_aug". <br />
To rename all the images in a folder from abc-->xyz run the following commands.
```
sudo apt-get install rename
rename 's/abc/xyz/' *
```
The above command will rename abc-->xyz.
- Change the src = "the source filepath to the dataset that is to be augmented" <br />
- Change save_prefix= "whatever prefix preferred for a specific class". It is recommended to use different prefixes for different image classes.

<br /> <br />
The [final model](https://github.com/TonduriGopinath7/construction-stage-detection/blob/master/Walls/model.h5) and it's [weights](https://github.com/TonduriGopinath7/construction-stage-detection/blob/master/Walls/cnn_model.hdf5) have also been uploaded. <br />

Run all these codes and you have a cnn model that predicts the stage of the walls with a 95% accuracy! :)
