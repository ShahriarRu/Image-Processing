import cv2 as cv

img = cv.imread(r'C:\Users\Asus\OneDrive\Documents\MyWebProjects\Image Processing\Practice\Photos\cat_large.jpg')

cv.imshow('Cat', img)
cv.waitKey(0)