import cv2
import numpy as np
import yaml

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


ref_img = cv2.imread("images/horarios.png")
raw = cv2.imread("images/horarios-perspective.png")
frame = camara_undistort(raw, mtx, dist)

ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

cv2.imshow('ref_gray', ref_gray)
cv2.imshow('frame_gray', frame_gray)

# Find size of ref_gray
sz = ref_gray.shape
warp_mode = cv2.MOTION_HOMOGRAPHY
warp_matrix = np.eye(3, 3, dtype=np.float32)
number_of_iterations = 5000
termination_eps = 1e-20
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

(cc, warp_matrix) = cv2.findTransformECC(ref_gray, frame_gray, warp_matrix, warp_mode, criteria, inputMask=None, gaussFiltSize=1)

frame_aligned = cv2.warpPerspective(frame, warp_matrix, (sz[1], sz[0]),flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

cv2.imshow('aligned', frame_aligned)
cv2.waitKey(0)

