#! /usr/bin/env python
# coding: utf-8
# uso: ./face-detection.py

############################################################################
# Visión por Computador. Curso 2019-20.
# Proyecto final: detección de caras a partir de una red YOLO preentrenada.
# Miguel Lentisco Ballesteros. Antonio Coín Castro.
############################################################################

#
# LIBRERÍAS
#

# Generales
import numpy as np

# Visualización
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Keras
import keras
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

# Utilidades
from bbox import *

#
# PARÁMETROS GLOBALES
#

N = 61                    # Número de clases
TAM = (10, 5)             # Tamaño del plot
PATH = "../data/"         # Directorio de trabajo
INPUT_SHAPE = (416, 416)  # Tamaño de las imágenes de entrada. Debe ser múltiplo de 32.

#
# FUNCIONES AUXILIARES
#

def wait():
    """Introduce una espera hasta que se pulse una tecla."""

    input("(Pulsa cualquier tecla para continuar...)")

#
# LECTURA Y PREPROCESADO DE IMÁGENES
#

def preprocess_input(x):
    """Realiza el preprocesamiento necesario para pasar cada imagen 'x'
       por la red YOLOv3."""

    return x.astype(np.float32) / 255.0

#
# PREDICCIÓN
#

def predict(model, flow, l):
    """Predicción de etiquetas sobre un conjunto de imágenes
       a partir de un flujo de imágenes.
        - model: modelo a usar para predecir.
        - flow: generador de imágenes.
        - l: longitud del conjunto de imágenes."""

    return model.predict_generator(flow, steps = l, verbose = 1)

#
# DETECCIÓN DE CARAS
#

def detect_one(yolo, filename):
    """Realiza detección de caras en una imagen. Muestra las
       regiones detectadas en la imagen original.
        - yolo: modelo YOLOv3.
        - filename: ruta relativa de la imagen."""

    # Leemos la imagen, guardando el tamaño original
    im_w, im_h = load_img(PATH + filename).size
    im = img_to_array(load_img(PATH + filename, target_size = INPUT_SHAPE))

    # Preprocesamos la imagen y añadimos una dimensión
    im = np.expand_dims(preprocess_input(im), 0)

    # Realizamos la predicción
    preds = yolo.predict(im)

    # define the anchors (determine with kmeans for new dataset)
    anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]
    # define the probability threshold for detected objects
    class_threshold = 0.6
    boxes = list()
    for i in range(len(preds)):
    	# decode the output of the network
    	boxes += decode_netout(preds[i][0], anchors[i], class_threshold, *INPUT_SHAPE)

    # correct the sizes of the bounding boxes for the shape of the image
    correct_yolo_boxes(boxes, im_h, im_w, *INPUT_SHAPE)

    # suppress non-maximal boxes
    do_nms(boxes, 0.5)

    # define the labels
    labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
        "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
        "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
        "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
        "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

    # get the details of the detected objects
    v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)

    #print
    for i in range(len(v_boxes)):
        print(v_labels[i], v_scores[i])

    # draw what we found
    draw_boxes(PATH + filename, v_boxes, v_labels, v_scores)

def face_detection():
    """Realiza detección de caras en el conjunto de test de la base de
       datos WIDERFACE."""

    # Creamos un datagen para preprocesar y generar las imágenes
    datagen = ImageDataGenerator(preprocessing_function = preprocess_input)

    # Creamos un flujo para ir leyendo las imágenes desde directorio
    dataflow = datagen.flow_from_directory(directory = PATH + 'test',
                                           target_size = SHAPE,
                                           class_mode = None,
                                           batch_size = 1,
                                           shuffle = False)

#
# FUNCIÓN PRINCIPAL
#

def main():
    """Ejecuta el programa."""

    print("\n--- DETECCIÓN DE CARAS CON YOLOv3 ---\n")

    # Cargamos el modelo YOLOv3
    yolo = load_model('yolov3.h5')

    # Realizamos la detección
    detect_one(yolo, 'zebra.jpg')

if __name__ == "__main__":
 main()
