import cv2
import numpy as np

width, height = 500, 500
counter = 0

def obtain_h_matrix(img):
  # declaro array vacio donde voy a guardar los puntos
  circles = np.zeros((4, 2), np.int)

  def mousePoints(event, x, y, flags, params):
    global counter
    if event == cv2.EVENT_LBUTTONDOWN:
      circles[counter] = x, y
      counter = counter + 1

  while True:
    if counter == 4:
      origin = np.float32([circles[0], circles[1], circles[2], circles[3]])
      target = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
      matrix = cv2.getPerspectiveTransform(origin, target)
      cv2.destroyWindow("Calibration Image ")
      return matrix

    for x in range(0, counter):
      cv2.circle(img, (circles[x][0], circles[x][1]), 3, (0, 255, 0), cv2.FILLED)

    cv2.imshow("Calibration Image ", img)
    cv2.setMouseCallback("Calibration Image ", mousePoints)
    cv2.waitKey(1)
