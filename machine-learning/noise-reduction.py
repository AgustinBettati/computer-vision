import cv2
import numpy as np

image = cv2.imread("/Users/abettati/projects/computer-vision/machine-learning/shapes/testing/rect-camara.png")
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
block_size = 67
bin = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, 2)

# Invert the image so the area of the UAV is filled with 1's. This is necessary since
# cv::findContours describes the boundary of areas consisting of 1's.
bin = 255 - bin

cv2.imshow('binary', bin)

kernel = np.ones((3,3),np.uint8)
# buscamos eliminar falsos positivos (puntos blancos en el fondo)
bin = cv2.morphologyEx(bin, cv2.MORPH_ERODE, kernel)
cv2.imshow('after', bin)

contours, hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
shape_contour = max(contours, key=cv2.contourArea)
cv2.drawContours(image, [shape_contour], -1, (255, 0, 255), 3)
cv2.imshow('with contours', image)
cv2.waitKey(0)
