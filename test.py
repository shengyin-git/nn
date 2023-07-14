## setup
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
# from tensorflow import keras
from pathlib import Path
from tensorflow.keras import applications
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from tensorflow.keras import Model
# from tensorflow.keras.applications import resnet
from tensorflow.keras.applications.resnet50 import ResNet50

## data: cats and dogs
# load data

# split train validate & test

# pre-process data


## hyperparameters
epochs = 10
batch_size = 16
# margin = 1  # Margin for contrastive loss.
target_shape = (300, 300)


## define the model
# input = layers.Input((200, 200, 3))
# x = tf.keras.layers.BatchNormalization()(input)
# x = layers.Conv2D(4, (5, 5), activation="tanh")(x)
# x = layers.AveragePooling2D(pool_size=(2, 2))(x)
# x = layers.Conv2D(16, (5, 5), activation="tanh")(x)
# x = layers.AveragePooling2D(pool_size=(2, 2))(x)
# x = layers.Flatten()(x)

# x = tf.keras.layers.BatchNormalization()(x)
# x = layers.Dense(10, activation="tanh")(x)
# embedding_network = keras.Model(input, x)

# base_cnn = resnet.ResNet50(
#     weights="imagenet", input_shape=target_shape + (3,), include_top=False
# )

# flatten = layers.Flatten()(base_cnn.output)
# dense1 = layers.Dense(512, activation="relu")(flatten)
# dense1 = layers.BatchNormalization()(dense1)
# dense2 = layers.Dense(256, activation="relu")(dense1)
# dense2 = layers.BatchNormalization()(dense2)
# output = layers.Dense(256)(dense2)

# embedding = Model(base_cnn.input, output, name="Embedding")

# input = layers.Input(target_shape + (3,)) # (200, 200, 3)

# base_cnn = resnet.ResNet50(
#     weights="imagenet", input_shape=target_shape + (3,), include_top=False
# )

# dnn_model = Sequential()
# imported_model = tf.keras.applications.ResNet50(include_top=False, input_shape=target_shape + (3,), pooling='avg', classes=5, weights='imagenet')
# for layer in imported_model.layers:
#     layer.trainable=False

# embedding_network = keras.Model(input, x)

# my model 
base_cnn = ResNet50(weights="imagenet", input_shape=target_shape + (3,), include_top=False)
output = resnet.layers[-1].output
output = layers.Flatten()(output)
# output = layers.Flatten()(base_cnn.output)

# trainable = False
# for layer in base_cnn.layers:
#     if layer.name == "conv5_block1_out":
#         trainable = True
#     layer.trainable = trainable

embedding = Model(base_cnn.input, output, name="Embedding")

for layer in base_cnn.layers:
    layer.trainable = False

embedding.summary()

















