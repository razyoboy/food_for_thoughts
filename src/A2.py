import tensorflow as tf
import numpy as np

from PIL import Image
from pathlib import Path

# Load 54 classes model
ML_MODEL = tf.keras.models.load_model(Path("model/ML_model.hdf5").__str__())

food_list = np.sort(['bibimbap','caesar_salad', 'cheesecake','chicken_curry','chicken_wings','chocolate_cake','club_sandwich','crab_cakes','creme_brulee','cup_cakes',\
             'donuts','dumplings','edamame','eggs_benedict','filet_mignon','fish_and_chips','foie_gras','french_fries','fried_rice','frozen_yogurt','garlic_bread',\
             'grilled_cheese_sandwich','grilled_salmon','gyoza','hamburger','hot_and_sour_soup','hot_dog','ice_cream','lasagna','lobster_bisque','macaroni_and_cheese',\
             'macarons','miso_soup','mussels','omelette','onion_rings','oysters','pad_thai','pancakes','panna_cotta','peking_duck','pho','pizza','ramen','steak',\
             'risotto','sashimi','scallops','spaghetti_bolognese','spaghetti_carbonara','sushi','takoyaki','tiramisu','waffles'])

classes = list(food_list)

def local_get(path):
    img = Image.open(path)

    # Resize img to proper for feed model.
    img = img.resize((299,299))
    
    # Convert img to numpy array,rescale it,expand dims and check vertically.
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = x / 255.0 
    x = np.expand_dims(x,axis = 0)
    img_tensor = np.vstack([x])
    
    return img_tensor

def predict_result(path):
    img_tensor = local_get(path)
    
    pred = ML_MODEL.predict(img_tensor)
    classes = list(food_list)
    class_predicted = classes[np.argmax(pred)]
    percent = np.max(pred)

    return percent, class_predicted


if __name__ == "__main__":
    confidence, label = predict_result(Path("img/donut.jpg").__str__())
    print(confidence)
    print(label)
