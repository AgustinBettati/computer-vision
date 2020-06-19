import yaml
import numpy as np
import cv2

# buscar archivo que tiene matriz
a_yaml_file = open("../calibration_matrix.yaml")
parsed_yaml_file = yaml.load(a_yaml_file, Loader=yaml.FullLoader)
mtx = np.array(parsed_yaml_file["camera_matrix"])
dist = np.array(parsed_yaml_file["dist_coeff"])

def camara_undistort(img):
  h, w = img.shape[:2]
  newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
  dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
  # crop the image
  x, y, w, h = roi
  return dst[y:y+h, x:x+w]
