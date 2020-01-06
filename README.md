# Detección de objetos usando una red YOLO preentrenada.

Utilizamos la red [YOLOv3](https://github.com/experiencor/keras-yolo3) implementada en Keras y preentrenada sobre la base de datos [COCO](http://cocodataset.org/#home) para detección de caras en la base de datos [WIDERFACE](http://shuoyang1213.me/WIDERFACE/).

Realizado junto a [@MiguelLentisco](https://github.com/MiguelLentisco). Curso 2019-20.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/antcc/proyecto-vc/blob/master/yolov3.ipynb)

## Dependencias

### Código

- Python 3.6.9
- TensorFlow 1.15+
- Keras 2.2.5+
- NumPy 1.18.0+
- Matplotlib 3.1.2+
- OpenCV 4.1.2+
- Pillow 7.0.0+

### Documentación

- Pandoc 2.7.3+
- Filtros de Pandoc: [pandoc-citeproc](https://github.com/jgm/pandoc-citeproc) + [pandoc-crossref](https://github.com/lierdakil/pandoc-crossref)

## Resultados

Hemos entrenado el modelo sobre el conjunto de entrenamiento proporcionado. La métrica de evaluación utilizada es la *mean average precission* ó *mAP* ([COCO Challenge 2017](http://cocodataset.org/#detection-eval)) sobre el conjunto de validación. Para calcular este valor se ha utilizado el servidor de evaluación disponible en la [competición de Codalab sobre WIDERFACE](https://competitions.codalab.org/competitions/20146).

| Épocas de entrenamiento | mAP     |
|:-----------------------:|:-------:|
| 100                     | 0.3     |
