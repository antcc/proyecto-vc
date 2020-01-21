# Introducción

Dataset WIDERFACE [@yang2016wider].

# Cosas por hacer

**Varios:**

- Entender todo el código. Eliminar lo que no sea necesario.
- Adaptar código para poder evaluar el conjunto de test (a partir de filelist, sin anotaciones).
- Ver por qué no coincide la métrica de evaluación de `evaluate_coco` con la de `codalab`. Reimplementar para que coincidan. Posiblemente cambiar el cálculo de AP a la interpolación en 101 pasos (https://kharshit.github.io/blog/2019/09/20/evaluation-metrics-for-object-detection-and-segmentation). Si no funciona, probar con 11 pasos.

**Entrenamiento:**

- Hacer finetuning a partir de los pesos de COCO iniciales (congelar unas cuantas capas, ¿cuáles?)
- Cambiar optimizador a SGD, RMSProp, Adabound(https://github.com/Luolc/AdaBound/blob/master/adabound/adabound.py) <---
- Entrenar más épocas. <---
- Entrenar con un valor mayor de *xywh_scale* en el config. Por ejemplo 2? <---
- Entrenar con un mayor tamaño de entrada de las imágenes. Ahora mismo en Colab no es viable.
- Aumentar el umbral *ignore_thresh*, por ejemplo a 0.6 ó 0.7.

**Evaluación:**

- Aumentar tamaño de entrada (no sé si tiene sentido que supere al input_size de entrenamiento) <---
- Aumentar umbral supresión de no máximos, por ejemplo a 0.6

**Opcionales:**

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

*** Entrenamiento tras 130 épocas: ***

Parámetros: min-input: 288, max-input: 512

AP (Pascal VOC 2007): ~0.68
mAP (COCO 2017): 0.38

*** Finetuning ***

--> 288,512,ig06,xywh2

- 17 épocas (early stopping) congelando todo menos las 10 últimas capas. Lr inicial de 1e-3, batch size de 12 (train_hist_finetuning). Loss: 53.42
- 80 épocas. Todo descongelado. Batch size 8. Lr inicial 1e-4.

mAP: 0.39

--> 352,512,ig07

- 30 épocas congelando todo menos los 3 bloques de detección. Lr inicial de 1e-3. Batch_size de 12. Warmup epochs = 4.
Loss: 37.72
Logs: finetuning-30
Tiempo estimado por época: 400s

- 50 épocas con todo descongelado. Lr inicial 1e-4. Batch size de 8.
Loss: 22.2
Logs:
Tiempo estimado por época: 700s

- ?

Evaluación:

input_size=1024, obj_thresh = 0.6, nms_thresh = 0.45

mAP:
AP:

-->416,672,ig07,xywh2

- 30 épocas congelando todo menos los 3 bloques de detección. Lr inicial 1e-3. Batch_size de 8. Warmup 4
Loss: 36
Logs:
tiempo estimado por época: 650s

- 70 épocas todo descongelado, 416,512, bs8,lr1e-4
Loss: 24
Tiempo:
evaluación
mAP: 0.4053
AP: 0.7155

*** Modelo base ***

- Cargando backend.h5 tal cual y haciendo finetune 10 épocas (para la última capa de cada bloque de detección).
Parámetros: input 416, ig0.5, min-max 416,416, obj0.5, nms0.45, jitter 0.0, xywh1, lr 1e-3
mAP@.5:.05:.95: 0.0181
AP@0.5: 0.0818


# Apéndice: Funcionamiento del código {.unnumbered}

<!-- Esto es una prueba de referencia al apéndice: [Apéndice A: Funcionamiento del código].-->

# Bibliografía {.unnumbered}
