import cv2
import yaml
import numpy
a_yaml_file = open("../calibration_matrix.yaml")
parsed_yaml_file = yaml.load(a_yaml_file, Loader=yaml.FullLoader)
print(parsed_yaml_file)

mtx = numpy.array(parsed_yaml_file["camera_matrix"])
dist = numpy.array(parsed_yaml_file["dist_coeff"])

def camara_undistort(img, mtx, dist):
  h, w = img.shape[:2]
  newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
  dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
  # crop the image
  x, y, w, h = roi
  return dst[y:y+h, x:x+w]

img = cv2.imread('images/ejemplo.png')
dst = camara_undistort(img, mtx, dist)
cv2.imwrite('calibresult.png', dst)



