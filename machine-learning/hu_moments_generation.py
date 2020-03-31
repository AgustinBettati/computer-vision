import cv2
import glob
import csv
import numpy
import math

def hu_moments_of_file(filename):
    im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    _, im = cv2.threshold(im, 128, 255, cv2.THRESH_BINARY)
    # Calculate Moments
    moments = cv2.moments(im)
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


with open('./generated-files/shapes-hu-moments.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    write_hu_moments("5-point-star", writer)
    write_hu_moments("rectangle", writer)
    write_hu_moments("triangle", writer)


