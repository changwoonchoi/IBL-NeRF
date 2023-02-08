import os
import cv2
import matplotlib.pyplot as plt
import numpy as np


def show_image():
	basedir = "../../data/mitsuba/kitchen"
	split="train"
	image_file_path = os.path.join(basedir, split, "1_albedo.png")
	image = cv2.imread(image_file_path)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = image.astype(np.float32)
	image /= 255.0
	plt.figure()
	plt.imshow(image)
	rgb2srgb=np.power(image, 1/2.2)
	plt.figure()
	plt.imshow(rgb2srgb)
	plt.show()


show_image()