import tensorflow as tf
import numpy as np

from PIL import Image
from pathlib import Path

# Load 54 classes model
ML_MODEL = tf.keras.models.load_model(Path("model/ML_model.hdf5").__str__())

food_list = ['Apple_Pie','Baby_Back_Ribs','Baklava','Beef_Carpaccio','Beef_Tartare','Beet_Salad','Beignets','Bibimbap','Boopadpongali','Bread_Pudding','Breakfast_Burrito', 
             'Bruschetta','Caesar_Salad','Cannoli','Caprese_Salad','Carrot_Cake','Ceviche','Cheese_Plate','Cheesecake','Chicken_Curry','Chicken_Quesadilla','Chicken_Wings', 
             'Chocolate_Cake','Chocolate_Mousse','Churros','Clam_Chowder','Club_Sandwich','Crab_Cakes','Creme_Brulee','Croque_Madame','Cup_Cakes','Curriedfishcake', 
             'Deviled_Eggs','Donuts','Dumplings','Edamame','Eggs_Benedict','Eggsstewed','Escargots','Falafel','Filet_Mignon','Fish_And_Chips','Foie_Gras','French_Fries', 
             'French_Onion_Soup','French_Toast','Fried_Calamari','Fried_Rice','Friedkale','Frozen_Yogurt','Gaengjued','Gaengkeawwan','Garlic_Bread','Gnocchi','Goongobwoonsen', 
             'Goongpao','Greek_Salad','Grilled_Cheese_Sandwich','Grilled_Salmon','Grilledqquid','Guacamole','Gyoza','Hamburger','Hot_And_Sour_Soup','Hot_Dog','Hoykraeng',
             'Hoylaiprikpao','Huevos_Rancheros','Hummus','Ice_Cream','Joke','Kaithoon','Kaomangai','Kaomoodang','Khanomjeennamyakati','Khaomokgai','Khaomootodgratiem', 
             'Khaoniewmamuang','Kkaoklukkaphi','Kormooyang','Kuakling','Kuayjab','Kuayteowreua','Larbmoo','Lasagna','Lobster_Bisque','Lobster_Roll_Sandwich','Macaroni_And_Cheese', 
             'Macarons','Massamangai','Miso_Soup','Moosatay','Mussels','Nachos','Namtokmoo','Omelette','Onion_Rings','Oysters','Pad_Thai','Padpakbung','Padpakruammit', 
             'Paella','Pancakes','Panna_Cotta','Peking_Duck','Phatkaphrao','Pho','Pizza','Pork_Chop','Porkstickynoodles','Poutine','Prime_Rib','Pulled_Pork_Sandwich', 
             'Ramen','Ravioli','Red_Velvet_Cake','Risotto','Roast_Duck','Roast_Fish','Samosa','Sashimi','Scallops','Seaweed_Salad','Shrimp_And_Grits','Somtam', 
             'Soninlaweggs','Spaghetti_Bolognese','Spaghetti_Carbonara','Spring_Rolls','Steak','Stewedporkleg','Strawberry_Shortcake','Sushi','Tacos','Takoyaki', 
             'Tiramisu','Tomkhagai','Tomyumgoong','Tuna_Tartare','Waffles','Yamwoonsen','Yentafo']

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
