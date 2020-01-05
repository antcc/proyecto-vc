# Introducción

Dataset WIDERFACE [@yang2016wider].

- The output of the model is, in fact, encoded candidate bounding boxes from three different grid sizes: 13x13, 26x26 y 52x52.

- TODO: implementar faster nms (https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/) 

PASOS SEGUIDOS:

1. Convertir las anotaciones de WIDERFACE a formato VOC. Para ello se ha usado el archivo `convert.py`, adaptado de https://github.com/akofman/wider-face-pascal-voc-annotations/blob/master/convert.py
4. Generar anchor boxes para nuestro conjunto usando k-means con gen-anchors.py, y ponerlos en el config.
5. Descargar los pesos backend.h5 preentrenados en COCO.
6. Comenzar el entrenamiento en nuestro conjunto.

*** Entrenamiento tras 100 épocas: ***

Parámetros: min-input: 288, max-input: 512

loss: 19.5426 - yolo_layer_1_loss: 0.8661 - yolo_layer_2_loss: 5.1030 - yolo_layer_3_loss: 13.5735

AP: 0.4739

**Evaluación:**

- Explicación de mAP: https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173

*** Cosas a probar para evaluar ***

- Cambiar umbral supresión de no máximos
- Cambiar umbral obj_thresh a 0.6 !!!
- Cambair tamaño de entrada (aumentar?)

# Apéndice: Funcionamiento del código {.unnumbered}

<!-- Esto es una prueba de referencia al apéndice: [Apéndice A: Funcionamiento del código].-->

# Bibliografía {.unnumbered}
