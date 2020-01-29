from PIL import Image 
import numpy as np 
import urllib.request
import h5py
import requests
from io import BytesIO
import matplotlib.pyplot as plt

def test(code, i):
	url = "http://s3.amazonaws.com/stardustathome.testbucket/real/{x}/{x}-001.jpg".format(x=code)
	r = requests.get(url)
	try:
		img = Image.open(BytesIO(r.content))
		img = np.array(img) / 255.0
	except OSError:
		print("got error from URL")
		img = np.ones((384, 512, 1))
	plt.imshow(img, cmap = 'gray')
	plt.title(code)
	plt.ion()
	plt.show(block = False)
	plt.waitforbuttonpress(30)
	plt.close()

def test_codes(code_file, start):
	i = start
	for code in code_file.read().splitlines()[start:]:
		test(code, i)
		i += 1

yes_codes = open("verified_codes.txt", "r")
test_codes(yes_codes, 120)