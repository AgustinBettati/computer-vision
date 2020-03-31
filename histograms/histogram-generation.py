import cv2
from matplotlib import pyplot as plt

#reading image as grayscale
img = cv2.imread('./foto.jpg',0)
plt.hist(img.ravel(),256,[0,256]); plt.show()
