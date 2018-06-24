# import the necessary packages
from PIL import Image
import pytesseract
import argparse
import cv2
import os
import numpy as np
from skimage.feature import match_template
import matplotlib.pyplot as plt

orig_gray = None
resultImage = None
x = None
y = None
levelarea = None

def getLevel(imagepath, username):
	result = [0,False]
	# load the example image and convert it to grayscale
	image = cv2.imread(imagepath)
	template =  cv2.imread("template_ios.PNG")
	gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	global orig_gray
	orig_gray = gray
	gray_template = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)

	# find location of level
	global resultImage
	resultImage = match_template(gray, gray_template)
	ij = np.unravel_index(np.argmax(resultImage), resultImage.shape)
	global x
	global y
	x, y = ij[::-1]

	# extract level from image
	height, width  = gray.shape
	# print("width: " + str(width) +"  "+ str(int(0.1*width)))
	global levelarea
	levelarea = gray[y-int(0.1*width):y+int(0.05*width),x-int(0.03*width):x+int(0.12*width)]
	level = pytesseract.image_to_string(levelarea)
	print("Found Level:")
	reqLevel = [int(s) for s in level.split() if s.isdigit()]
	if reqLevel == []:
		print("no Level found")
		result[0] = None
	else:
		print(reqLevel[0])
		result[0] = reqLevel[0]


	# check to see if we should apply thresholding to preprocess the
	# image
	gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	# grey = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

	# write the grayscale image to disk as a temporary file so we can
	# apply OCR to it
	filename = "{}.png".format(os.getpid())
	cv2.imwrite(filename, gray)

	# load the image as a PIL/Pillow image, apply OCR, and then delete
	# the temporary file
	text = pytesseract.image_to_string(Image.open(filename))
	os.remove(filename)
	# print(text)
	clearText = ''.join(e for e in text if e.isalnum())
	if (clearText.find(username) >= 0):
		print("Username found: " + username)
		result[1] = True
	else:
		print("Username not found")
		result[1] = False

	return result


if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required=True,
		help="path to input image to be OCR'd")
	ap.add_argument("-u", "--username", required=True,
		help="username to search in OCR output")
	ap.add_argument("-p", "--plot", type=str, default="false",
		help="true, to display plot, false else")
	args = vars(ap.parse_args())

	getLevel(args["image"],args["username"])

	# plott images
	if (args["plot"] == "true"):
		fig = plt.figure(figsize=(8, 3))
		ax1 = plt.subplot(1, 3, 1)
		ax2 = plt.subplot(1, 3, 2)
		ax3 = plt.subplot(1, 3, 3, sharex=ax2, sharey=ax2)
		ax1.imshow(orig_gray)
		ax2.imshow(levelarea)
		ax3.imshow(resultImage)
		ax3.set_axis_off()
		ax3.set_title('`match_template`\nresult')
		# highlight matched region
		ax3.autoscale(False)
		ax3.plot(x, y, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)

		plt.show()
