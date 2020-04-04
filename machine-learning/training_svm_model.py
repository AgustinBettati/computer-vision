import cv2
import csv

import numpy as np
from label_converters import label_to_int

trainData = []
trainLabels = []

def load_training_set():
    global trainData
    global trainLabels
    with open('./generated-files/shapes-hu-moments.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            class_label = row.pop()
            floats = list(map(lambda n: float(n), row))
            trainData.append(np.array(floats, dtype=np.float32))
            trainLabels.append(np.array([label_to_int(class_label)], dtype=np.int32))
    trainData = np.array(trainData, dtype=np.float32)
    trainLabels = np.array(trainLabels, dtype=np.int32)

def train_svm_model():
    load_training_set()

    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_RBF)
    svm.trainAuto(trainData, cv2.ml.ROW_SAMPLE, trainLabels)
    svm.save("./generated-files/svm_shapes_model.yml")

# testSample = np.array([trainData[15]], dtype=np.float32)
# testResponse = svm.predict(testSample)[1]
# print(testResponse)
