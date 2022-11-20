import tensorflow as tf
import matplotlib.image as img
import numpy as np
from collections import defaultdict
import collections
from shutil import copy
from shutil import copytree, rmtree
import matplotlib.pyplot as plt
import os
import random
from tensorflow import keras
import cv2 as cv
import pathlib

# data_dir = pathlib.Path("./dataset/food/images/")
# all_classes = len(list(data_dir.glob("*")))

"""Split the image into train and test"""
# def prepare_data(filepath, src,dest):
#   classes_images = defaultdict(list)
#   with open(filepath, 'r') as txt:
#       paths = [read.strip() for read in txt.readlines()]
#       for p in paths:
#         food = p.split('/')
#         classes_images[food[0]].append(food[1] + '.jpg')

#   for food in classes_images.keys():
#     print("\nCopying images into ",food)
#     if not os.path.exists(os.path.join(dest,food)):
#       os.makedirs(os.path.join(dest,food))
#     for i in classes_images[food]:
#       copy(os.path.join(src,food,i), os.path.join(dest,food,i))
#   print(data_dir,)
#============= Run the fucntion ================================
# prepare_data("dataset/food/meta/meta/train.txt", data_dir, 'trainer')
# prepare_data("dataset/food/meta/meta/test.txt", data_dir, 'tester')


"""Choose the set of images to train"""
# def dataset_mini(food_list, src, dest):
#   if os.path.exists(dest):
#     rmtree(dest) # removing dataset_mini(if it already exists) folders so that we will have only the classes that we want
#   os.makedirs(dest)
#   for food_item in food_list :
#     print("Copying images into",food_item)
#     copytree(os.path.join(src,food_item), os.path.join(dest,food_item))

# food_list = ['pad_thai', 'ice_cream', 'fried_rice']
# # src_test = 'subDataset/tester'
# # dst_test = 'subDataset/subTest'
# src_train = 'subDataset/trainer'
# dst_train = 'subDataset/subTrain'
#============= Run the fucntion ================================
# dataset_mini(food_list, src_train, dst_train)

train_folder = pathlib.Path("./subDataset/subTrain")
test_folder = pathlib.Path("./subDataset/subTest")

classes = len(list(train_folder.glob("*")))
width, height = 299, 299
batch_size = 16
train_samples = len(list(train_folder.glob("*/*")))
test_samples = len(list(test_folder.glob("*/*")))

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)


test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(
    train_folder,
    target_size=(height, width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    test_folder,
    target_size=(height, width),
    batch_size=batch_size,
    class_mode='categorical')

inception = tf.keras.applications.inception_v3.InceptionV3(weights='imagenet', include_top=False)
x = inception.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128,activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)

predictions = tf.keras.layers.Dense(3,kernel_regularizer=tf.keras.regularizers.l2(0.005), activation='softmax')(x)

model = tf.keras.models.Model(inputs=inception.input, outputs=predictions)
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath='best_model_3class.hdf5', verbose=1, save_best_only=True)
csv_logger = tf.keras.callbacks.CSVLogger('history_3class.log')

history = model.fit(train_generator,
                    steps_per_epoch = train_samples // batch_size,
                    validation_data=validation_generator,
                    validation_steps=test_samples // batch_size,
                    epochs=30,
                    verbose=1,
                    callbacks=[csv_logger, checkpointer])

model.save('model_trained_3class.hdf5')
