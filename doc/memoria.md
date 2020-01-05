# Introducción

Dataset WIDERFACE [@yang2016wider].


- Evaluation:  If the ratio of the intersection of a detected region with an annotated face region is greater than 0.5, a score of 1 is assigned to the detected region, and 0 otherwise.

- The output of the model is, in fact, encoded candidate bounding boxes from three different grid sizes: 13x13, 26x26 y 52x52.

- TODO: cambiar la resolución de las imagenes de entrada (aumentarlas) para ver si mejora, a 600 y pico u 800 (múltiplos de 32).

PASOS SEGUIDOS:

1. Convertir las anotaciones de WIDERFACE a formato VOC. Para ello se ha usado el archivo `convert.py`, adaptado de https://github.com/akofman/wider-face-pascal-voc-annotations/blob/master/convert.py (editado para las anotaciones inválidas: 0 0 0 0 0 0 0). Editar `voc.py` para poner 'path' en vez de 'filename' al leer las anotaciones.
2. Adaptar el código de https://github.com/experiencor/keras-yolo3 a tensorflow v2 mediante la utilidad tf_upgrade_v2.
3. Configurar el config.json para nuestro dataset. Explicar cada uno de los parámetros.
4. Generar anchor boxes para nuestro conjunto usando k-means con gen-anchors.py, y ponerlos en el config.
5. Descargar los pesos backend.h5 () preentrenados en COCO (creo, espero, no lo pone por ningún sitio).
6. Comenzar el entrenamiento en nuestro conjunto.

*** Entrenamiento tras 50 épocas: ***

Parámetros: min-input: 288, max-input: 512

loss: 20.6447 - yolo_layer_1_loss: 1.0436 - yolo_layer_2_loss: 5.6852 - yolo_layer_3_loss: 13.9159

mAP: 0.4659

*** Entrenamiento tras 100 épocas: ***

Parámetros: min-input: 288, max-input: 512

loss: 19.5426 - yolo_layer_1_loss: 0.8661 - yolo_layer_2_loss: 5.1030 - yolo_layer_3_loss: 13.5735

mAP: 0.4739

<!-- Esto es una prueba de referencia [@test] y esto es otra al apéndice: [Apéndice A: Funcionamiento del código].-->

# Apéndice: Funcionamiento del código {.unnumbered}

# Bibliografía {.unnumbered}
