import cv2
import numpy as np

from hu_moments_generation import *

def int_to_label(string_label):
  if string_label == 1: return '5-point-star'
  if string_label == 2: return 'rectangle'
  if string_label == 3: return 'triangle'
  else:
    raise Exception('unkown class_label')

svm = cv2.ml.SVM_load('./generated-files/svm_shapes_model.yml')
print(svm)

files = glob.glob('./shapes/testing/*')
for f in files:
    hu_moments = hu_moments_of_file(f)
    sample = np.array([hu_moments], dtype=np.float32)
    testResponse = svm.predict(sample)[1]
    print(f + ' -> ' + int_to_label(testResponse))
