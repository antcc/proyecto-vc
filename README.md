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

### Documentación

- Pandoc 2.7.3+
- Filtros de Pandoc: [pandoc-citeproc](https://github.com/jgm/pandoc-citeproc) + [pandoc-crossref](https://github.com/lierdakil/pandoc-crossref)

## Resultados

Hemos entrenado el modelo sobre el conjunto de entrenamiento proporcionado. La métrica de evaluación utilizada es la [*mean average precission*](http://cocodataset.org/#detection-eval) sobre el conjunto de validación. Para calcular este valor se ha utilizado el servidor de evaluación disponible en la [competición de Codalab sobre WIDERFACE](https://competitions.codalab.org/competitions/20146).

| Modelo                              | Épocas de entrenamiento | Tamaño de entrada | mAP@.5:.05:.95     | mAP@0.5  |
|:-----------------------------------:|:-----------------------:|:-----------------:|:------------------:|:--------:|
| Entrenamiento completo              | 130                     | 416x416           | 0.3                | 0.4739   |
| Entrenamiento completo              | 130                     | 1024x1024         | 0.385              | 0.68     |
| Finetuning + entrenamiento completo | 75                      | 1024x1024         | 0.3923             | 0.7082   |
| Finetuning + entrenamiento completo | 100                     | 1024x1024         | 0.4053             | 0.7255   |
