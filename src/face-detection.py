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
import matplotlib.pyplot as plt

# Keras y TensorFlow
import keras
import keras.utils as np_utils
from tensorflow.compat.v1 import logging

# Modelos y capas
from keras.models import Model, Sequential

# Lectura y preprocesamiento de datos
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator

#
# PARÁMETROS GLOBALES
#

N = 200                  # Número de clases
TAM = (10, 5)            # Tamaño del plot
PATH = "./"              # Directorio de trabajo

#
# FUNCIONES AUXILIARES
#

def wait():
    """Introduce una espera hasta que se pulse una tecla."""

    input("(Pulsa cualquier tecla para continuar...)")

#
# LECTURA Y MODIFICACIÓN DEL CONJUNTO DE IMÁGENES
#

def read_im(names):
    """Lee las imágenes cuyos nombres están especificados en un vector de entrada.
       Devuelve las imágenes en un vector y sus clases en otro.
        - names: vector con los nombres (ruta relativa) de las imágenes."""

    classes = np.array([im.split('/')[0] for im in names])
    vim = np.array([img_to_array(load_img(PATH + im, target_size = INPUT_SIZE))
                    for im in names])

    return vim, classes

def load_data():
    """Carga el conjunto de datos en 4 vectores: las imágenes de entrenamiento,
       las clases de las imágenes de entrenamiento, las imágenes de test y las
       clases de las imágenes de test.

       Lee las imágenes y las clases de los ficheros 'train.txt' y 'test.txt'."""

    # Cargamos los ficheros
    train_images = np.loadtxt(PATH + "train.txt", dtype = str)
    test_images = np.loadtxt(PATH + "test.txt", dtype = str)

    # Leemos las imágenes
    train, train_classes = read_im(train_images)
    test, test_classes = read_im(test_images)

    # Convertimos las clases a números enteros
    unique_classes = np.unique(np.copy(train_classes))
    for i in range(len(unique_classes)):
      train_classes[train_classes == unique_classes[i]] = i
      test_classes[test_classes == unique_classes[i]] = i

    # Convertimos los vectores de clases en matrices binarias
    train_classes = np_utils.to_categorical(train_classes, N)
    test_classes = np_utils.to_categorical(test_classes, N)

    # Barajamos los datos
    train_perm = np.random.permutation(len(train))
    train = train[train_perm]
    train_classes = train_classes[train_perm]

    test_perm = np.random.permutation(len(test))
    test = test[test_perm]
    test_classes = test_classes[test_perm]

    return train, train_classes, test, test_classes

#
# ESTADÍSTICAS
#

def show_stats(score, hist, name, show = True):
    """Muestra estadísticas de accuracy y loss y gráficas de evolución.
        - score: métricas de evaluación.
        - hist: historial de entrenamiento.
        - name: nombre del modelo.
        - show: controla si se muestran gráficas con estadísticas."""

    print("\n---------- " + name.upper() + " EVALUATION ----------")
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    print()

#
# PREDICIÓN Y EVALUACIÓN
#

def predict_gen(model, datagen, x):
    """Predicción de etiquetas sobre un conjunto de imágenes.
        - model: modelo a usar para predecir.
        - datagen: generador de imágenes.
        - x: conjunto de datos para predecir su clase."""

    preds = model.predict_generator(datagen.flow(x,
                                                 batch_size = 1,
                                                 shuffle = False),
                                    verbose = 1,
                                    steps = len(x))

    return preds

def evaluate(model, x_test, y_test):
    """Evaluar el modelo sobre el conjunto de test.
        - model: modelo a usar para evaluar.
        - x_test, y_test: datos de test."""

    score = model.evaluate(x_test, y_test, verbose = 0)

    return score

def evaluate_gen(model, datagen, x_test, y_test):
    """Evaluar el modelo sobre el conjunto de test, usando un generador
       de imágenes.
        - model: modelo a usar para evaluar.
        - datagen: generador de imágenes.
        - x_test, y_test: datos de test."""

    score = model.evaluate_generator(datagen.flow(x_test,
                                                  y_test,
                                                  batch_size = 1,
                                                  shuffle = False),
                                      verbose = 0,
                                      steps = len(x_test))

    return score

#
# FUNCIÓN PRINCIPAL
#

def main():
    """Ejecuta el programa."""

    # No mostrar warnings de TensorFlow
    logging.set_verbosity(logging.ERROR)

    print("\n--- DETECCIÓN DE CARAS CON YOLOv3 ---\n")

if __name__ == "__main__":
 main()
