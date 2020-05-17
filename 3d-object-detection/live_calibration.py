import math

import cv2
import numpy as np
import yaml

width, height = 500, 500
counter = 0
cap = cv2.VideoCapture(0)

a_yaml_file = open("../calibration_matrix.yaml")
parsed_yaml_file = yaml.load(a_yaml_file, Loader=yaml.FullLoader)
mtx = np.array(parsed_yaml_file["camera_matrix"])
dist = np.array(parsed_yaml_file["dist_coeff"])

def camara_undistort(img, mtx, dist):
  h, w = img.shape[:2]
  newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
  dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
  # crop the image
  x, y, w, h = roi
  return dst[y:y+h, x:x+w]

def obtain_h_matrix(img):
  circles = np.zeros((4, 2), np.int)

  def mousePoints(event, x, y, flags, params):
    global counter
    if event == cv2.EVENT_LBUTTONDOWN:
      circles[counter] = x, y
      counter = counter + 1

  while True:
    if counter == 4:
      pts1 = np.float32([circles[0], circles[1], circles[2], circles[3]])
      pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
      matrix = cv2.getPerspectiveTransform(pts1, pts2)
      cv2.destroyWindow("Calibration Image ")
      return matrix

    for x in range(0, 4):
      cv2.circle(img, (circles[x][0], circles[x][1]), 3, (0, 255, 0), cv2.FILLED)

    cv2.imshow("Calibration Image ", img)
    cv2.setMouseCallback("Calibration Image ", mousePoints)
    cv2.waitKey(1)


global h_mat
global inverse_h_mat
h_mat = []
inverse_h_mat = []

def inverse_contour_points(shape_contour):
  new_contours = map(invert_point, shape_contour)
  # x = np.array([[cX, cY]], dtype='float32')
  # original_points = cv2.perspectiveTransform(x[np.newaxis], inverse_mat)
  # center = original_points[0][0]
  new_contours = np.array(list(new_contours))
  return new_contours

def invert_point(x):
  x = np.array(x, dtype='float32')
  original_points = cv2.perspectiveTransform(x[np.newaxis], inverse_mat)
  result = list(map(lambda x: np.array([math.floor(x[0]), math.floor(x[1])]),original_points[0]))
  result = np.array(result)
  return result


while True:

    ret, raw = cap.read()
    # raw = cv2.flip(raw, 1)
    frame = camara_undistort(raw, mtx, dist)

    if cv2.waitKey(1) == ord('c'):
        h_mat = obtain_h_matrix(frame)
        inverse_mat = cv2.invert(h_mat)[1]
        print(h_mat)

    if(len(h_mat) != 0):
        imgOutput = cv2.warpPerspective(frame, h_mat, ((width, height)))

        gray = cv2.cvtColor(imgOutput, cv2.COLOR_RGB2GRAY)
        block_size = 67  # Tamaño del bloque a comparar, debe ser impar.
        bin = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, 2)

        # Invert the image so the area of the UAV is filled with 1's. This is necessary since
        # cv::findContours describes the boundary of areas consisting of 1's.
        bin = 255 - bin  # como sabemos que las figuras son negras invertimos los valores binarios para que esten en 1.

        kernel = np.ones((3, 3), np.uint8)  # Tamaño del bloque a recorrer
        # buscamos eliminar falsos positivos (puntos blancos en el fondo) para eliminar ruido.
        bin = cv2.morphologyEx(bin, cv2.MORPH_ERODE, kernel)

        contours, hierarchy = cv2.findContours(bin, cv2.RETR_LIST,
                                               cv2.CHAIN_APPROX_SIMPLE)  # encuetra los contornos, chain aprox simple une algunos puntos para que no sea discontinuo.
        shape_contour = max(contours, key=cv2.contourArea)  # Agarra el contorno de area maxima
        if(cv2.contourArea(shape_contour) > 100):
            cv2.drawContours(imgOutput, [shape_contour], -1, (255, 0, 255), 3)

            original_contours = inverse_contour_points(shape_contour)
            cv2.drawContours(frame, [original_contours], -1, (255, 0, 255), 3)


            M = cv2.moments(shape_contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(imgOutput, (cX, cY), 7, (255, 255, 255), -1)
            cv2.putText(imgOutput, "(" + str(cX) + ", " + str(cY) + ")", (cX - 20, cY - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            x = np.array([[cX, cY]], dtype='float32')
            original_points = cv2.perspectiveTransform(x[np.newaxis], inverse_mat)
            center = original_points[0][0]
            cv2.circle(frame, (math.floor(center[0]), math.floor(center[1])), 7, (255, 255, 255), -1)
            cv2.putText(frame, "(" + str(math.floor(center[0])) + ", " + str(math.floor(center[1])) + ")", (math.floor(center[0]) - 20, math.floor(center[1]) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


        cv2.imshow("2D Image ", imgOutput)

    cv2.imshow("Live Image ", frame)


cap.release()
cv2.destroyAllWindows()

