from os import listdir
from os.path import isfile, join
from PIL import Image 
import numpy as np 
import urllib.request
import h5py
import requests
from io import BytesIO
import matplotlib.pyplot as plt

def save(code):
	url = "http://s3.amazonaws.com/stardustathome.testbucket/real/{x}/{x}-001.jpg".format(x=code)
	r = requests.get(url)
	try:
		img = Image.open(BytesIO(r.content))
		img = np.array(img) / 255.0
		plt.imshow(img, cmap = 'gray')
		plt.title(code)
		plt.axis("off")
		plt.savefig("/Users/loganjaeger/Desktop/stardust/Deployment/Possible Craters/" + code + ".png")
		plt.close()
	except OSError:
		print(url)

def get_codes(path):
	onlyfiles = [f for f in listdir(path)]
	for f in onlyfiles:
		f = f[:-4]
		save(f)

p = "/Users/loganjaeger/Desktop/stardust/Deployment/Best Possible Craters"
get_codes(p)