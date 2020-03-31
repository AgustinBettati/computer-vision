import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('./foto.jpg')
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

color = ('r','g')
for i,col in enumerate(color):
    histr = cv.calcHist([hsv],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()
