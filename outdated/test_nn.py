import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
from pathlib import Path
from tensorflow.keras import applications
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import Model
from tensorflow.keras.applications.resnet50 import ResNet50
import glob
from tensorflow.keras.preprocessing.image import ImageDataGenerator, apply_affine_transform, random_rotation, random_shift, load_img, img_to_array, array_to_img
# import pandas as pd 
# import shutil
import time
from tensorflow.python.ops.numpy_ops import np_config

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())
# from keras import backend as K
# K.tensorflow_backend._get_available_gpus()

## hyperparameters
epochs = 10
batch_size = 16
target_shape = (300, 300)

## data: cats and dogs
# !!!!!!!!!!!!!!!!!! data set from https://www.kaggle.com/c/dogs-vs-cats/data
# load data and split train validate & test
files = glob.glob('./data/train/*')
cat_files = [fn for fn in files if 'cat' in fn]
dog_files = [fn for fn in files if 'dog' in fn]
len(cat_files)
len(dog_files)

cat_train = np.random.choice(cat_files, size=1500, replace=False)
dog_train = np.random.choice(dog_files, size=1500, replace=False)

cat_files = list(set(cat_files)-set(cat_train))
dog_files = list(set(dog_files)-set(dog_train))

cat_val = np.random.choice(cat_files, size=500, replace=False)
dog_val = np.random.choice(dog_files, size=500, replace=False)

cat_files = list(set(cat_files)-set(cat_val))
dog_files = list(set(dog_files)-set(dog_val))

cat_test = np.random.choice(cat_files, size=500, replace=False)
dog_test = np.random.choice(dog_files, size=500, replace=False)

print('cat datasets:', cat_train.shape, cat_val.shape, cat_test.shape)
print('dog datasets:', dog_train.shape, dog_val.shape, dog_test.shape)

cat_train_imgs = [img_to_array(load_img(img, target_size=target_shape)) for img in cat_train]
dog_train_imgs = [img_to_array(load_img(img, target_size=target_shape)) for img in dog_train]

cat_val_imgs = [img_to_array(load_img(img, target_size=target_shape)) for img in cat_val]
dog_val_imgs = [img_to_array(load_img(img, target_size=target_shape)) for img in dog_val]

cat_test_imgs = [img_to_array(load_img(img, target_size=target_shape)) for img in cat_test]
dog_test_imgs = [img_to_array(load_img(img, target_size=target_shape)) for img in dog_test]

cat_train_imgs = np.array(cat_train_imgs)
cat_train_imgs_scaled = cat_train_imgs.astype('float32')
cat_train_imgs_scaled /= 255

dog_train_imgs = np.array(dog_train_imgs)
dog_train_imgs_scaled = dog_train_imgs.astype('float32')
dog_train_imgs_scaled /= 255

cat_val_imgs = np.array(cat_val_imgs)
cat_val_imgs_scaled = cat_val_imgs.astype('float32')
cat_val_imgs_scaled /= 255

dog_val_imgs = np.array(dog_val_imgs)
dog_val_imgs_scaled = dog_val_imgs.astype('float32')
dog_val_imgs_scaled /= 255

cat_test_imgs = np.array(cat_test_imgs)
cat_test_imgs_scaled = cat_test_imgs.astype('float32')
cat_test_imgs_scaled /= 255

dog_test_imgs = np.array(dog_test_imgs)
dog_test_imgs_scaled = dog_test_imgs.astype('float32')
dog_test_imgs_scaled /= 255

# show the example pic and conduct rotation and transformation

fig, ax = plt.subplots(4,5,figsize=(15,6))
l = [ax[0,i].imshow(cat_train_imgs_scaled[i]) for i in range(5)]

for i in range(5):
    # y1 = apply_affine_transform(x=cat_train_imgs_scaled[i], theta=45, tx=0, ty=0, shear=0, zx=1, zy=1, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.0, order=1)
    y1 = apply_affine_transform(x=cat_train_imgs_scaled[i], theta=45, row_axis=0, col_axis=1, channel_axis=2, fill_mode='constant', cval=1.0)
    ax[1,i].imshow(y1) 
    y2 = apply_affine_transform(x=cat_train_imgs_scaled[i], tx=100, row_axis=0, col_axis=1, channel_axis=2, fill_mode='constant', cval=1.0)
    ax[2,i].imshow(y2) 
    y3 = apply_affine_transform(x=cat_train_imgs_scaled[i], ty=100, row_axis=0, col_axis=1, channel_axis=2, fill_mode='constant', cval=1.0)
    ax[3,i].imshow(y3) 

    ts = time.gmtime()
    ts = time.strftime("%Y_%m_%d_%H_%M_%S", ts)
    plt.savefig('./data/example_rotation_and_translation_' + ts + '.png')

# plt.show()

# input('example code for image rotation and translation.')

# for i in range(5):
#     layer_rotate = layers.RandomRotation(factor=(0.25/2,0.25/2), fill_mode='constant', fill_value=1.0,)
#     rotate = layer_rotate(cat_train_imgs_scaled[i])
#     rotate = rotate.astype('float32')
#     ax[1,i].imshow(rotate) 

# for i in range(5):
#     layer_rotate = layers.RandomTranslation(height_factor=(0.25/2,0.25/2), width_factor=(0,0), fill_mode='constant', fill_value=1.0,)
#     rotate = layer_rotate(cat_train_imgs_scaled[i])
#     rotate = rotate.astype('float32')
#     ax[2,i].imshow(rotate) 

# for i in range(5):
#     layer_rotate = layers.RandomTranslation(height_factor=(0,0), width_factor=(0.25/2,0.25/2), fill_mode='constant', fill_value=1.0,)
#     rotate = layer_rotate(cat_train_imgs_scaled[i])
#     rotate = rotate.astype('float32')
#     ax[3,i].imshow(rotate) 
# plt.show() 

# datagen = ImageDataGenerator()
#     rotate = datagen.apply_transform(x=cat_train_imgs[i], transform_parameters={'ty':100})
#     rotate = random_shift(x=cat_train_imgs[i], wrg=[100,100], hrg= [0,0], fill_mode='constant')

# train_datagen = ImageDataGenerator(rescale=1, zoom_range=1, rotation_range=50,\
#     width_shift_range=0.2, height_shift_range=0.2, shear_range=0, \
#     horizontal_flip=False, fill_mode='nearest')
# val_datagen = ImageDataGenerator(rescale=1)

# img_id = 500
# cat_generator = train_datagen.flow(cat_train_imgs[img_id:img_id+1], batch_size=1) 
# cat = [next(cat_generator) for i in range(5)] 
# fig, ax = plt.subplots(1,5, figsize=(16, 6))
# print('Labels:', [item[1][0] for item in cat]) 
# l = [ax[i].imshow(cat[i][0][0]) for i in range(0,5)]
# plt.show()

# (x_train_val, y_train_val), (x_test, y_test) = keras.datasets.mnist.load_data()

# pre-process data
def make_pairs(x, y):
    """Creates a tuple containing image pairs with corresponding label.
    """
    _num = len(x)
    _index = np.arange(_num)
    pairs = []
    labels = []
    for i in range(_num):
        # add a matching example
        z1 = x[i]
        z2 = x[random.choice(_index)]
        pairs += [[z1, z2]]
        labels += [0]

        z1 = y[i]
        z2 = y[random.choice(_index)]
        pairs += [[z1, z2]]
        labels += [0]

        # add a non-matching example
        z1 = x[i]
        z2 = y[random.choice(_index)]
        pairs += [[z1, z2]]
        labels += [1]

        z1 = y[i]
        z2 = x[random.choice(_index)]
        pairs += [[z1, z2]]
        labels += [1]

    return np.array(pairs), np.array(labels).astype("float32")

# make train pairs
pairs_train, labels_train = make_pairs(cat_train_imgs_scaled, dog_train_imgs_scaled)

# make validation pairs
pairs_val, labels_val = make_pairs(cat_val_imgs_scaled, dog_val_imgs_scaled)

# make test pairs
pairs_test, labels_test = make_pairs(cat_test_imgs_scaled, dog_test_imgs_scaled)

# Split the training pairs
x_train_1 = pairs_train[:, 0]  # x_train_1.shape is (60000, 28, 28)
x_train_2 = pairs_train[:, 1]

x_val_1 = pairs_val[:, 0]  # x_val_1.shape = (60000, 28, 28)
x_val_2 = pairs_val[:, 1]

x_test_1 = pairs_test[:, 0]  # x_test_1.shape = (20000, 28, 28)
x_test_2 = pairs_test[:, 1]

## visualize pairs and their labels
def visualize(pairs, labels, to_show=6, num_col=3, predictions=None, test=False):
    """Creates a plot of pairs and labels, and prediction if it's test dataset.

    Arguments:
        pairs: Numpy Array, of pairs to visualize, having shape
               (Number of pairs, 2, 28, 28).
        to_show: Int, number of examples to visualize (default is 6)
                `to_show` must be an integral multiple of `num_col`.
                 Otherwise it will be trimmed if it is greater than num_col,
                 and incremented if if it is less then num_col.
        num_col: Int, number of images in one row - (default is 3)
                 For test and train respectively, it should not exceed 3 and 7.
        predictions: Numpy Array of predictions with shape (to_show, 1) -
                     (default is None)
                     Must be passed when test=True.
        test: Boolean telling whether the dataset being visualized is
              train dataset or test dataset - (default False).

    Returns:
        None.
    """

    # Define num_row
    # If to_show % num_col != 0
    #    trim to_show,
    #       to trim to_show limit num_row to the point where
    #       to_show % num_col == 0
    #
    # If to_show//num_col == 0
    #    then it means num_col is greater then to_show
    #    increment to_show
    #       to increment to_show set num_row to 1
    num_row = to_show // num_col if to_show // num_col != 0 else 1

    # `to_show` must be an integral multiple of `num_col`
    #  we found num_row and we have num_col
    #  to increment or decrement to_show
    #  to make it integral multiple of `num_col`
    #  simply set it equal to num_row * num_col
    to_show = num_row * num_col

    # Plot the images
    fig, axes = plt.subplots(num_row, num_col, figsize=(10, 10))
    for i in range(to_show):

        # If the number of rows is 1, the axes array is one-dimensional
        if num_row == 1:
            ax = axes[i % num_col]
        else:
            ax = axes[i // num_col, i % num_col]

        ax.imshow(tf.concat([pairs[i][0], pairs[i][1]], axis=1), cmap="gray")
        ax.set_axis_off()
        if test:
            ax.set_title("True: {} | Pred: {:.5f}".format(labels[i], predictions[i][0]))
        else:
            ax.set_title("Label: {}".format(labels[i]))
    # if test:
    #     plt.tight_layout(rect=(0, 0, 1.9, 1.9), w_pad=0.0)
    # else:
    #     plt.tight_layout(rect=(0, 0, 1.5, 1.5))
    
    ts = time.gmtime()
    ts = time.strftime("%Y_%m_%d_%H_%M_%S", ts)
    plt.savefig('./data/example_training_pairs_' + ts + '.png')

    # plt.show()

visualize(pairs_train[:-1], labels_train[:-1], to_show=3)

## define the model
# extract features using trained resnet
base_cnn = ResNet50(weights="imagenet", input_shape=target_shape + (3,), include_top=False)
output = base_cnn.layers[-1].output
output = layers.Flatten()(output)

embedding_network = Model(base_cnn.input, output, name="Embedding")

for layer in base_cnn.layers:
    layer.trainable = False

embedding_network.summary()

input_1 = layers.Input(target_shape + (3,))
input_2 = layers.Input(target_shape + (3,))

tower_1 = embedding_network(input_1)
tower_2 = embedding_network(input_2)

# merge features
merge_layer = layers.Concatenate()([tower_1, tower_2])

# classification based on merged fetures
dense_layer_0 = layers.Dense(1024, activation='relu')(merge_layer)
drop_out_layer_0 = layers.Dropout(0.3)(dense_layer_0)
dense_layer_1 = layers.Dense(512, activation='relu')(drop_out_layer_0)
drop_out_layer_1 = layers.Dropout(0.3)(dense_layer_1)
# normal_layer = layers.BatchNormalization()(merge_layer)
output_layer = layers.Dense(1, activation="sigmoid")(drop_out_layer_1)

# nn model
nn = Model(inputs=[input_1, input_2], outputs=output_layer)

nn.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=1e-5), metrics=['accuracy'])

nn.summary()

## train the model
history = nn.fit(
    [x_train_1, x_train_2],
    labels_train,
    validation_data=([x_val_1, x_val_2], labels_val),
    batch_size=batch_size,
    epochs=epochs,
)

## visualization
def plt_metric(history, metric, title, has_valid=True):
    """Plots the given 'metric' from 'history'.

    Arguments:
        history: history attribute of History object returned from Model.fit.
        metric: Metric to plot, a string value present as key in 'history'.
        title: A string to be used as title of plot.
        has_valid: Boolean, true if valid data was passed to Model.fit else false.

    Returns:
        None.
    """
    fig = plt.figure()
    plt.plot(history[metric])
    if has_valid:
        plt.plot(history["val_" + metric])
        plt.legend(["train", "validation"], loc="upper left")
    plt.title(title)
    plt.ylabel(metric)
    plt.xlabel("epoch")

    ts = time.gmtime()
    ts = time.strftime("%Y_%m_%d_%H_%M_%S", ts)
    plt.savefig('./data/example_testing_pairs_' + ts + '.png')

    # plt.show()

# Plot the accuracy
plt_metric(history=history.history, metric="accuracy", title="Model Accuracy")

# Plot the contrastive loss
plt_metric(history=history.history, metric="loss", title="Binary Crossentropy Loss")

## test the model
results = nn.evaluate([x_test_1, x_test_2], labels_test)
print("test loss, test acc:", results)

predictions = nn.predict([x_test_1, x_test_2])
visualize(pairs_test, labels_test, to_show=3, predictions=predictions, test=True)

input()






















