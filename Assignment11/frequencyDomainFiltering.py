import matplotlib.pyplot as plt
import cv2
import numpy as np

def main():
	# Load image.
	img_path = 'images/cube.jpg'#GhostPepper.jpg' #PaddyField.jpeg'
	rgb = plt.imread(img_path)

	# Convert images
	gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

	# Perform Fast Fourier Transformation for 2D signal, i.e., image
	ftimg = np.fft.fft2(gray)
	centered_ftimg = np.fft.fftshift(ftimg)
	magnitude_spectrum = 100 * np.log(np.abs(ftimg))
	centered_magnitude_spectrum = 100 * np.log(np.abs(centered_ftimg))

	width,height = gray.shape
	black = np.zeros((width,height), dtype = 'uint8')
	cv2.circle(black, (black.shape[1]//2 , black.shape[0]//2), 80, (1,1,1), thickness=-1)

	gaussian = cv2.GaussianBlur(gray,(5,5),cv2.BORDER_DEFAULT)

	gaussian_mask = gaussian*black
	
	
	# Apply Gaussian filter
	ftimg_gf1 = centered_ftimg * gaussian_mask
	filtered_img1 = np.abs(np.fft.ifft2(ftimg_gf1))

	kernel = np.array([[2, 0, -2], [1, 0, -1], [2, 0, -2]])
	Filter1 = cv2.filter2D(gray,-1,kernel)
	filter_mask1 = Filter1 * black

	ftimg_gf2 = centered_ftimg * filter_mask1
	filtered_img2 = np.abs(np.fft.ifft2(ftimg_gf2))

	# Save images all together by matplotlib.
	img_set = [rgb, gray, magnitude_spectrum, centered_magnitude_spectrum, gaussian_mask, filtered_img1, filter_mask1,filtered_img2]
	title_set = ['RGB', 'Gray', 'FFT2', 'Centered FFT2', 'Gaussian Filter', 'Filtered Img', "Random Filter", "Random Effect"]
	matplotlib_plot_img(img_set, title_set)
	
def matplotlib_plot_img(img_set, title_set):
	plt.rcParams.update({'font.size': 16})			
	plt.figure(figsize = (10, 6))
	n = len(img_set)
	for i in range(n):
		plt.subplot(2, 4, i + 1)
		plt.title(title_set[i])
		img = img_set[i]
		ch = len(img.shape)
		if (ch == 2):
			plt.imshow(img, cmap = 'gray')
		else:
			plt.imshow(img)			

	plt.tight_layout()
	plt.savefig("output.png")

if __name__ == '__main__':
	main()