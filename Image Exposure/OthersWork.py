#	============================================================
#	Purpose: To investigate over-exposed and under-exposed
#	channels.
#	------------------------------------------------------------
#	Sangeeta Biswas
#	Assistant Professor
#	University of Rajshahi, Rajshahi
#	16.7.2021
#	============================================================

import matplotlib.pyplot as plt
import cv2
import numpy as np

DIR = '/home/bibrity/Retina_Research/Exposure_Problem/System/'

def main():
	img_path = DIR + 'Data/OverExposure2.jpeg'
	rgb = plt.imread(img_path)
	hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
	gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
	CIELab =  cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
	print(CIELab.shape)
	
	imgset = [rgb, gray, rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2], hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2], CIELab[:, :, 0], CIELab[:, :, 1], CIELab[:, :, 2]]
	titleset = ['RGB', 'Gray', 'Red', 'Green', 'Blue', 'Hue', 'Saturation', 'Value', 'L*', 'a*', 'b*']
	plot_images(imgset, titleset)
	
def plot_images(imgset, titleset):
	n = len(imgset)
	plt.figure(figsize = (20,20))
	for i in range(n):
		plt.subplot(3, 4, i + 1)
		plt.title(titleset[i])
		plt.imshow(imgset[i], cmap = 'gray')
		plt.axis('off')
	
	plt.show()
	plt.close()

if __name__ == '__main__':
	main()
