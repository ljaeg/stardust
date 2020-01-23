#This is a deployment method biased against false positives. As Zack says, instant science
#This will HOPEFULLY give me a few false positives and a couple of true positives

#load the models for the different sizes
model150 = ...
model100 = ...
model30 = ...

#specify the thresholds for the different sizes
th_150 = .5
th_100 = .7
th_30 = .8

NumImages = 10000 #number of images to look through

#We need to normalize the image in some way
def norm1(im):
	#just put between 0 and 1
	mn = np.min(im)
	mx = np.max(im)
	return (im - mn) / (mx - mn)

def norm2(im):
	#just mean subtraction
	mean = np.mean(im)
	return im - mean 

def norm3(im):
	#mean subtraction and std of 1
	mean = np.mean(im)
	std = np.std(im)
	return (im - mean) / std 

def split_predict_150(im):
	a = [0, 100, 200, 234]
	b = [0, 100, 200, 300, 362]
	lower_preds = [0]
	for i in a:
		w = i + 150
		for j in b:
			z = j + 150
			sub_img = (image[i:w, j:z]).reshape(1, 150, 150, 1)
			sin = norm1(sub_img)
			pred = model.predict(sin)
			if pred > th_150:
				new_pred = split_predict_100(sin)
				lower_preds.append(new_pred)
	return max(lower_preds)

def split_predict_100(im):
	#takes a 150x150 img and predicts on it using 100x100 imgs
	a = [0, 50]
	pred_30s = [0]
	for i in a:
		w = i + 100
		for j in a:
			z = j + 100
			sub_img = im[i:w, j:z].reshape(1, 100, 100, 1)
			sin = norm1(sub_img)
			pred = model100.predict(sin)
			if pred > th_100:
				pred30 = split_predict_30(sin)
				pred_30s.append(pred30)
	return max(pred_30s)

def split_predict_30(im):
	a = [0, 15, 30, 45, 60, 70]
	preds = []
	for i in a:
		w = i + 30
		for j in a:
			z = j + 30
			sub_img = im[i:w, j:z].reshape(1, 30, 30, 1)
			sin = norm1(sub_img)
			pred = model30.predict(sin)
			if pred > th_30:
				#early cutoff
				return 1
	return 0

def is_control(im):
	s = im.shape
	if s[0] == 385 or s[0] == 513:
		return 1
	else:
		return 0
	

for i in range(NumImages):
	#load image
	#get image ID, maybe URL
	img = ...
	if is_control(im):
		#do the same thing but add to a control group
		#go to next iteration
	pred = split_predict_150(img)
	if pred == 1:
		#add url to txt file
		"""
		then I'll have a program that goes thru the txt file,
		shows me, a human, the image, and I'll determine if it 
		has a crater or not. Boo ya
		"""


















