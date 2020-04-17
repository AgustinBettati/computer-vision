import cv2
import numpy as np
import glob


from hu_moments_generation import hu_moments_of_file
from label_converters import int_to_label

def load_and_test(show_contours):
    # model = cv2.ml.SVM_load('./generated-files/svm_shapes_model.yml')
    # model = cv2.ml.NormalBayesClassifier_load('./generated-files/bayes_shapes_model.yml')
    model = cv2.ml.DTrees_load('./generated-files/tree_shapes_model.yml') #Carga el modelo que fue entrenado
    files = glob.glob('./shapes/testing/*')
    for f in files:
        hu_moments = hu_moments_of_file(f, show_contours) #Genera los momentos de hu de los files de testing
        sample = np.array([hu_moments], dtype=np.float32) #numpy
        testResponse = model.predict(sample)[1] #Predice la clase de cada file
        print(f + ' -> ' + int_to_label(testResponse))

        #Lee la imagen y la imprime con un texto
        image = cv2.imread(f)
        image_with_text = cv2.putText(image, int_to_label(testResponse), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow("result", image_with_text)
        cv2.waitKey(0)
