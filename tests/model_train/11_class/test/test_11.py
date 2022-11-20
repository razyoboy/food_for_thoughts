import tensorflow as tf
import matplotlib.image as img
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import pathlib
from io import BytesIO
import requests
from PIL import Image
import os


food_list = ['caesar_salad', 'cheesecake', 'donuts', 'fish_and_chips', 'french_fries', 'fried_rice', 'hamburger', 'pad_thai', 'steak', 'sushi', 'waffles']
def predict_class(model, images, show = True):
  for img in images:
    img = tf.keras.preprocessing.image.load_img(img, target_size=(299, 299))
    img = tf.keras.preprocessing.image.img_to_array(img)                    
    img = np.expand_dims(img, axis=0)         
    img /= 255.                                      

    pred = model.predict(img)
    index = np.argmax(pred)
    food_list.sort()
    pred_value = food_list[index]
    
    if show:
        plt.imshow(img[0])                           
        plt.axis('off')
        plt.title(pred_value)
        plt.show()
    print(pred)

model_best = tf.keras.models.load_model('best_model_11class.hdf5',compile = False)

#==================Change your path here =======================
path = "/home/kjm/Documents/dataset/sub_food/9_class_test/"
img0 = path+"73180.jpg"
img1 = path+"cheesy.jpg"
img2 = path+"donut.jpg"
img3 = path+"fish_and_chips.jpg"
img4 = path+"Grazed-Donut.jpg"
img5 = path+"steak.jpg"
#================================================================

images = []
images.append(img0)
images.append(img1)
images.append(img2)
images.append(img3)
images.append(img4)
images.append(img5)

predict_class(model_best, images, True)