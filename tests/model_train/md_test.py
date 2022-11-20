import tensorflow as tf
import matplotlib.image as img
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

#To use this code, please install pip install pyqt6 on the environment
food_list = ['pad_thai', 'ice_cream', 'fried_rice']

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

model_best = tf.keras.models.load_model('./best_model_3class.hdf5',compile = False)
img0 = "./test_img/ice-cream-cake-11.jpg"
img1 = "./test_img/Pad_Thai.jpg"
img2 = "./test_img/rice.jpeg"

images = []
images.append(img0)
images.append(img1)
images.append(img2)

predict_class(model_best, images, True)