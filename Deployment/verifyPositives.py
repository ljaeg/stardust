#The pipeline has identified a few images as having craters
#I will now look over those and decide if any are right

from PIL import Image 
import numpy as np 
import urllib.request
import h5py
import requests
from io import BytesIO
import matplotlib.pyplot as plt

def test(code, f, super_f):
	url = "http://s3.amazonaws.com/stardustathome.testbucket/real/{x}/{x}-001.jpg".format(x=code)
	r = requests.get(url)
	try:
		img = Image.open(BytesIO(r.content))
		img = np.array(img) / 255.0
	except OSError:
		print("got error from URL")
		img = np.ones((384, 512, 1))
	plt.imshow(img, cmap = 'gray')
	plt.ion()
	plt.show(block = False)
	plt.waitforbuttonpress(25)
	plt.close()
	human_answer = input('Y or N: ')
	while not (human_answer == 'y' or human_answer == 'n' or human_answer == 'Y' or human_answer == 'N'):
		print('please enter valid answer')
		human_answer = input('Y or N: ')
	if human_answer == "Y" or human_answer == "y":
		f.write(code)
		f.write("\n")
	if human_answer == "yes":
		super_f.write(code)
		super_f.write("\n")

def test_codes(code_file):
	f = open("verified_codes.txt", "w")
	sf = open("super_verified_codes.txt", "w")
	for code in code_file.read().splitlines():
		test(code, f, sf)


#Note that I got thru approx the first 120 before, so adjust that for next time
yes_codes = open("/Users/loganjaeger/Desktop/stardust/yesCodes.txt", "r")
test_codes(yes_codes)