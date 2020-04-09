import cv2
import glob
import csv
import numpy
import math

def hu_moments_of_file(filename):
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    block_size = 67
    bin = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, 2)

    # Invert the image so the area of the UAV is filled with 1's. This is necessary since
    # cv::findContours describes the boundary of areas consisting of 1's.
    bin = 255 - bin

    kernel = numpy.ones((3, 3), numpy.uint8)
    # buscamos eliminar falsos positivos (puntos blancos en el fondo)
    bin = cv2.morphologyEx(bin, cv2.MORPH_ERODE, kernel)
    # cv2.imshow('binary', bin)
    contours, hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    shape_contour = max(contours, key=cv2.contourArea)
    # cv2.drawContours(image, [shape_contour], -1, (255, 0, 255), 3)
    # cv2.imshow('with contours', image)
    # cv2.waitKey(0)

    # Calculate Moments
    moments = cv2.moments(shape_contour)
    # Calculate Hu Moments
    huMoments = cv2.HuMoments(moments)
    # Log scale hu moments
    for i in range(0, 7):
        huMoments[i] = -1 * math.copysign(1.0, huMoments[i]) * math.log10(abs(huMoments[i]))
    return huMoments

def write_hu_moments(id, writer):
    files = glob.glob('./shapes/' + id + '/*')
    hu_moments = list(map(hu_moments_of_file, files))
    for mom in hu_moments:
      flattened = mom.ravel()
      row = numpy.append(flattened, id)
      writer.writerow(row)


def generate_hu_moments_file():
    with open('./generated-files/shapes-hu-moments.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        write_hu_moments("5-point-star", writer)
        write_hu_moments("rectangle", writer)
        write_hu_moments("triangle", writer)


generate_hu_moments_file()
