# Introducción

Dataset WIDERFACE [@yang2016wider].

# Cosas por hacer

**Varios:**

- Entender todo el código. Eliminar lo que no sea necesario.

**Entrenamiento:**

- Cambiar optimizador a SGD, RMSProp, Adabound(https://github.com/Luolc/AdaBound/blob/master/adabound/adabound.py) <---
- Entrenar más épocas. <---
- Entrenar con un valor mayor de *xywh_scale* en el config. Por ejemplo 3? <---
- Entrenar con un mayor tamaño de entrada de las imágenes. Ahora mismo en Colab no es viable.
- Aumentar el umbral *ignore_thresh*, por ejemplo a 0.6 ó 0.7.

**Evaluación:**

- Aumentar tamaño de entrada (no sé si tiene sentido que supere al input_size de entrenamiento) <---
- Aumentar umbral supresión de no máximos, por ejemplo a 0.6

**Opcionales:**

- Ver por qué no coincide la métrica de evaluación de `evaluate_coco` con la de `codalab`. Reimplementar para que coincidan.

- Reimplementar la funcíón de supresión de no máximos en su versión vectorizada, para que sea más rápida. Adaptar implementación de https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/

# Información a tener en cuenta

- The output of the model is, in fact, encoded candidate bounding boxes from three different grid sizes: 13x13, 26x26 y 52x52.

- Explicación de mAP: `https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173`

PASOS SEGUIDOS:

1. Convertir las anotaciones de WIDERFACE a formato VOC. Para ello se ha usado el archivo `convert.py`, adaptado de https://github.com/akofman/wider-face-pascal-voc-annotations/blob/master/convert.py
4. Generar anchor boxes para nuestro conjunto usando k-means con gen-anchors.py, y ponerlos en el config.
5. Descargar los pesos backend.h5 preentrenados en COCO.
6. Comenzar el entrenamiento en nuestro conjunto.
7. Validar usando el servidor de Codalab (https://competitions.codalab.org/competitions/2014).

*** Entrenamiento tras 100 épocas: ***

Parámetros: min-input: 288, max-input: 512

loss: 19.5426 - yolo_layer_1_loss: 0.8661 - yolo_layer_2_loss: 5.1030 - yolo_layer_3_loss: 13.5735
AP (Pascal VOC 2007): 0.4739
mAP (COCO 2017): 0.3

# Apéndice: Funcionamiento del código {.unnumbered}

<!-- Esto es una prueba de referencia al apéndice: [Apéndice A: Funcionamiento del código].-->

# Bibliografía {.unnumbered}
