This is an example neural network implementation for image comparison between cats and dogs.
Dataset can be accessed from https://www.kaggle.com/c/dogs-vs-cats/data.

The inputs include two images, if both cats the label is 0, if both dogs the label is 0, if different the label is 1.

This setting up is similar to what we want to do.

Our inputs include one image of masked items of one specific pile, and a mask image of the item of interest (triple the mask to make it compatible with resnet50 (like a rgb image))

The rotation and translation function are also included. 
