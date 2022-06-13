import cv2 as cv

img = cv.imread(r'C:\Users\Asus\OneDrive\Documents\MyWebProjects\Image Processing\Practice\Photos\cat_large.jpg')


def rescaleFrame(frame, scale=0.2):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (width,height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

resized_image = rescaleFrame(img)

cv.imshow('Cat', resized_image)
cv.waitKey(0)