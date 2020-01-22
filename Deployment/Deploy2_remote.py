"""
In this one, we will be doing web scraping.
We will go through images on the actual site, using the Machine account.
While we do this, we will answer whether or not a crater exists, as well
as save the images and the image ids to some database.
We will be using Selenium for browser manipulation.

The images on Stardust at home are 384x512, and we'll be using a NN designed for 150x150 images
"""

#import all the things
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
import numpy as np 
import tensorflow as tf 
import keras.backend as K
import h5py 
from keras.models import Sequential, load_model, Model
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
import tensorflow as tf
import urllib.request
from PIL import Image

#configure the GPU
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

#Paths
chrome_path = "/home/admin/Downloads/chromedriver"
#chrome_path = "/home/admin/Desktop/forChromeDriver/chromedriver"
img_path = "/home/admin/Desktop/GH/Deployment/Answers"


#import web scraping/interaction things
from selenium.webdriver import Chrome
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.keys import Keys
from pyvirtualdisplay import Display
display = Display(visible=0, size=(800, 800))  
display.start()
driver = Chrome(executable_path = chrome_path)

# Calculate the F1 score which we use for optimizing the CNN.
def f1_acc(y_true, y_pred):
    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))
    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0 or c2 == 0:
        return 0
    # How many selected items are relevant?
    precision = c1 / c2
    # How many relevant items are selected?
    recall = c1 / c3
    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

X_ims = 100
model = load_model('/home/admin/Desktop/Saved_CNNs/Foils_CNN_acc_FOV150.h5', custom_objects={'f1_acc': f1_acc})

#this is how to evaluate a single image
def split_image_and_pred(image):
	a = [0, 100, 200, 234]
	b = [0, 100, 200, 300, 362]
	all_preds = []
	for i in a:
		w = i + 150
		for j in b:
			z = j + 150
			sub_img = (image[i:w, j:z]).reshape(1, 150, 150, 1)
			pred = model.predict(sub_img)
			all_preds.append(pred)
	return max(all_preds)

#Navigate into the virtual microscope
driver.get("http://foils.ssl.berkeley.edu/index.php")
username = driver.find_element_by_name("name")
username.send_keys("Machine")
password = driver.find_element_by_name("password")
password.send_keys("23Na35Cl"+Keys.ENTER)
microscope = driver.find_element_by_partial_link_text("Virtual")
microscope.click()

#predict X_ims number of images
for _ in range(X_ims):
	img_element = driver.find_element_by_name("movieframe")
	img_url = img_element.get_attribute("src")
	movie_id = driver.find_element_by_xpath("//table[@class='body_12']/tbody/tr[1]/td[3]").text
	img = Image.open(urllib.request.urlopen(img_url))
	img_array = np.array(img) / 255
	print(img_array.shape)
	driver.close()
	"""
	if split_image_and_pred(img_array) > .5:
		img.save(img_path + "/positive/" + movie_id + ".png")
		img_element.click()
	else:
		img.save(img_path + "/negative/" +movie_id+ ".png")
		no_crater = driver.find_element_by_xpath("//img[@alt='No Good']/..")
		no_crater.click()




driver.close()


"""




