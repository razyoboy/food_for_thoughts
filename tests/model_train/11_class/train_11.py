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

#==========Please ignore this section if you already have the train/test images==================
# pfolder = "/home/kjm/Documents/dataset/food/images"
# ptext_train = "/home/kjm/Documents/dataset/food/meta/meta/train.txt" 
# ptext_test = "/home/kjm/Documents/dataset/food/meta/meta/test.txt" 

# data_dir = pathlib.Path(pfolder)
# all_classes = len(list(data_dir.glob("*")))

# """Split the image into train and test"""
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
# # ============= Run the fucntion ================================
# prepare_data(ptext_train, data_dir, 'pic/trainer')
# prepare_data(ptext_test, data_dir, 'pic/tester')
# # The image will show at the working directory
#=================================================================================================


"""Choose the set of images to train"""
def dataset_mini(food_list, src, dest):
  if os.path.exists(dest):
    rmtree(dest) # removing dataset_mini(if it already exists) folders so that we will have only the classes that we want
  os.makedirs(dest)
  for food_item in food_list :
    print("Copying images into",food_item)
    copytree(os.path.join(src,food_item), os.path.join(dest,food_item))

food_list = ['caesar_salad', 'cheesecake', 'donuts', 'fish_and_chips', 'french_fries', 'fried_rice', 'hamburger', 'pad_thai', 'steak', 'sushi', 'waffles']

# Please check that in your working directory have the these files or not. If not, please run the above section 
src_test = "pic/tester"
dst_test = "pic/split/test11"
src_train = "pic/trainer"
dst_train = "pic/split/train11"

# ============= Run the fucntion ================================
dataset_mini(food_list, src_train, dst_train)
dataset_mini(food_list, src_test, dst_test)

train_folder = pathlib.Path("pic/split/train11/")
test_folder = pathlib.Path("pic/split/test11/")

#==========Please change here regarding to your train model classe=============
n_classes=11 #this one
img_width, img_height = 299, 299
train_data_dir = 'pic/split/train11/'
validation_data_dir = 'pic/split/test11'
nb_train_samples = 8250 #this one
nb_validation_samples = 2750 #and this one
batch_size = 16
#==============================================================================
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')


inception = tf.keras.applications.inception_v3.InceptionV3(weights='imagenet', include_top=False)
x = inception.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(128,activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)

predictions = tf.keras.layers.Dense(n_classes,kernel_regularizer=tf.keras.regularizers.l2(0.005), activation='softmax')(x)

model = tf.keras.models.Model(inputs=inception.input, outputs=predictions)
model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath='test/best_model_11class.hdf5', verbose=1, save_best_only=True)
csv_logger = tf.keras.callbacks.CSVLogger('history_11class.log')
history_11class = model.fit_generator(train_generator,
                    steps_per_epoch = nb_train_samples // batch_size,
                    validation_data=validation_generator,
                    validation_steps=nb_validation_samples // batch_size,
                    epochs=30,
                    verbose=1,
                    callbacks=[csv_logger, checkpointer])
model.save('model_trained_11class.hdf5')