from PIL import Image 
import numpy as np 
import urllib.request
import h5py
import requests
from io import BytesIO
import matplotlib.pyplot as plt

def test(code):
	url = "http://s3.amazonaws.com/stardustathome.testbucket/real/{x}/{x}-001.jpg".format(x=code)
	r = requests.get(url)
	try:
		img = Image.open(BytesIO(r.content))
		img = np.array(img) / 255.0
	except OSError:
		print("got error from URL")
		return
	plt.imshow(img, cmap = 'gray')
	plt.title(code)
	plt.axis("off")
	# plt.ion()
	# plt.show(block = False)
	# plt.waitforbuttonpress()
	plt.savefig(code + ".png")
	plt.close()

def test_codes(code_file):
	for code in code_file.read().splitlines():
		test(code)

yes_codes = open("all_codes_batch1.txt", "r")
test_codes(yes_codes)