#The pipeline has identified a few images as having craters
#I will now look over those and decide if any are right

from PIL import Image 
import numpy as np 
import urllib.request
import h5py
import requests
from io import BytesIO
import matplotlib.pyplot as plt

def test(code, f, i):
	url = "http://s3.amazonaws.com/stardustathome.testbucket/real/{x}/{x}-001.jpg".format(x=code)
	r = requests.get(url)
	try:
		img = Image.open(BytesIO(r.content))
		img = np.array(img) / 255.0
	except OSError:
		print("got error from URL")
		img = np.ones((384, 512, 1))
	plt.imshow(img, cmap = 'gray')
	plt.title(code + " " + str(i))
	plt.ion()
	plt.show(block = False)
	plt.waitforbuttonpress(25)
	plt.close()
	human_answer = input('Y or N: ')
	while not (human_answer == 'y' or human_answer == 'n' or human_answer == 'Y' or human_answer == 'N' or human_answer == "end"):
		print('please enter valid answer')
		human_answer = input('Y or N: ')
	if human_answer == "Y" or human_answer == "y":
		f.write(code)
		f.write("\n")
		return 0
	if human_answer == "end":
		return 1
	else:
		return 0

def test_codes(code_file, start):
	f = open("verified_codes_2.txt", "a+")
	f1 = open("verified_codes.txt", "r")
	in_already = set()
	for i in f.read().splitlines():
		in_already.add(i)
	for i in f1.read().splitlines():
		in_already.add(i)
	i = start
	for code in code_file.read().splitlines()[start:]:
		if code in in_already:
			i+=1
			continue
		x = test(code, f, i)
		if x:
			break
		i += 1
	print(i)


#Note that I got thru approx the first 120 before, so adjust that for next time
yes_codes = open("/Users/loganjaeger/Desktop/stardust/yesCodes.txt", "r")
test_codes(yes_codes, 308)



