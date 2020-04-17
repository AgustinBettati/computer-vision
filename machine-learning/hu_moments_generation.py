import cv2
import glob
import csv
import numpy
import math

#Encargada de generar los momentos de Hu para las imagenes
def hu_moments_of_file(filename, show_contours):
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    block_size = 67 #Tamaño del bloque a comparar, debe ser impar.
    bin = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, 2)

    # Invert the image so the area of the UAV is filled with 1's. This is necessary since
    # cv::findContours describes the boundary of areas consisting of 1's.
    bin = 255 - bin # como sabemos que las figuras son negras invertimos los valores binarios para que esten en 1.

    kernel = numpy.ones((3, 3), numpy.uint8) #Tamaño del bloque a recorrer
    # buscamos eliminar falsos positivos (puntos blancos en el fondo) para eliminar ruido.
    bin = cv2.morphologyEx(bin, cv2.MORPH_ERODE, kernel)

    contours, hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) #encuetra los contornos, chain aprox simple une algunos puntos para que no sea discontinuo.
    shape_contour = max(contours, key=cv2.contourArea) #Agarra el contorno de area maxima
    if show_contours: #elijo si quiero mostrar o no los contornos desde el main.
        cv2.imshow('binary', bin)
        cv2.drawContours(image, [shape_contour], -1, (255, 0, 255), 3)
        cv2.imshow('with contours', image)
        cv2.waitKey(0)

    # Calculate Moments
    moments = cv2.moments(shape_contour) #momentos de inercia
    # Calculate Hu Moments
    huMoments = cv2.HuMoments(moments) #momentos de Hu
    # Log scale hu moments
    for i in range(0, 7):
        huMoments[i] = -1 * math.copysign(1.0, huMoments[i]) * math.log10(abs(huMoments[i])) #Mapeo para agrandar la escala.
    return huMoments


#Encargada de generar el archivo que contiene los valores de los momentos de Hu
def generate_hu_moments_file(show_contours):
    with open('./generated-files/shapes-hu-moments.csv', 'w', newline='') as file: #Se genera un archivo nuevo (W=Write)
        writer = csv.writer(file)
        #Ahora escribo los momentos de Hu de cada uno de las figuras. Con el string "rectangle...etc" busca en la carpeta donde estan cada una de las imagenes
        #generar los momentos de Hu y los escribe sobre este archivo. (LOS DE ENTRENAMIENTO).
        write_hu_moments("5-point-star", writer, show_contours)
        write_hu_moments("rectangle", writer, show_contours)
        write_hu_moments("triangle", writer, show_contours)


#Escribo los valores de los momentos de Hu en el archivo
def write_hu_moments(id, writer, show_contours):
    files = glob.glob('./shapes/' + id + '/*') #id recibe el nombre de la carpeta
    hu_moments = list(map(lambda f: hu_moments_of_file(f, show_contours), files)) #es un for que va sobre cada file, genera los momentos de hu para cada uno.
    for mom in hu_moments:
      flattened = mom.ravel()  #paso de un array de arrays a un array simple.
      row = numpy.append(flattened, id) # le metes el flattened array y le agregas el id
      writer.writerow(row) # Escribe una linea en el archivo.
