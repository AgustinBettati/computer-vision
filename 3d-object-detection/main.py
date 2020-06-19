import math

import cv2
import numpy as np

from h_matrix import obtain_h_matrix
from undistort import camara_undistort

width, height = 500, 500
width_cm, height_cm = 21, 21

cap = cv2.VideoCapture(0)

global h_mat
global inverse_h_mat
h_mat = []
inverse_h_mat = []

def inverse_contour_points(shape_contour):
  new_contours = map(invert_point, shape_contour)
  new_contours = np.array(list(new_contours))
  return new_contours

def invert_point(x):
  x = np.array(x, dtype='float32')
  original_points = cv2.perspectiveTransform(x[np.newaxis], inverse_mat)
  result = list(map(lambda x: np.array([math.floor(x[0]), math.floor(x[1])]),original_points[0]))
  result = np.array(result)
  return result

def draw_circle_with_text(img, x, y, label):
  cv2.circle(img, (x, y), 7, (255, 255, 255), -1)
  cv2.putText(img, label, (x - 20, y - 20),
              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
  cv2.putText(img, "(" + str(x) + ", " + str(y) + ")", (x - 20, y - 40),
              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

def obtain_contour(img):
  gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
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
  return max(contours, key=cv2.contourArea)  # Agarra el contorno de area maxima


def draw_axis(frame, perpendicular_img):
  draw_lines(perpendicular_img, (0,0),(width, 0), (0, height), (width - 50, 20), (20, height - 50))
  inverted_origin = invert_point([[0, 0]])[0]
  y_axis = invert_point([[0, height]])[0]
  x_axis = invert_point([[width, 0]])[0]
  y_text = invert_point([[20, height + 50]])[0]
  x_text = invert_point([[width + 50, 20]])[0]
  draw_lines(frame, to_tuple(inverted_origin),to_tuple(x_axis), to_tuple(y_axis), to_tuple(x_text), to_tuple(y_text))

def to_tuple(arr):
  return (arr[0], arr[1])

def draw_lines(img, origin, x_axis, y_axis, x_text, y_text):
  line_thickness = 4
  cv2.arrowedLine(img, origin, y_axis, (0, 255, 0), thickness=line_thickness)
  cv2.arrowedLine(img, origin, x_axis, (0, 255, 0), thickness=line_thickness)
  cv2.putText(img, "x", x_text,
              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
  cv2.putText(img, "y", y_text,
              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

while True:

    ret, raw = cap.read()
    frame = camara_undistort(raw)

    if cv2.waitKey(1) == ord('c'):
        h_mat = obtain_h_matrix(frame)
        inverse_mat = cv2.invert(h_mat)[1]

    if(len(h_mat) != 0):
        perpendicular_img = cv2.warpPerspective(frame, h_mat, ((width, height)))

        shape_contour = obtain_contour(perpendicular_img)
        if(cv2.contourArea(shape_contour) > 1000):
            cv2.drawContours(perpendicular_img, [shape_contour], -1, (255, 0, 255), 3)

            original_contours = inverse_contour_points(shape_contour)
            cv2.drawContours(frame, [original_contours], -1, (255, 0, 255), 3)

            M = cv2.moments(shape_contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            x_cm = width_cm * cX / width
            y_cm = height_cm * cY / height

            draw_circle_with_text(perpendicular_img, cX, cY, "(" + str(x_cm) + ", " + str(y_cm) + ") cm")

            inverted_point = invert_point([[cX, cY]])[0]
            draw_circle_with_text(frame, inverted_point[0], inverted_point[1], "(" + str(x_cm) + ", " + str(y_cm) + ") cm")

        draw_axis(frame, perpendicular_img)
        cv2.imshow("Homography ", perpendicular_img)

    cv2.imshow("Live Image ", frame)


cap.release()
cv2.destroyAllWindows()

