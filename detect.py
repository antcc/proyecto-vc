import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
from yolo.utils.bbox import draw_boxes, BoundBox
from keras.models import load_model
from yolo.utils.utils import get_yolo_boxes
from yolo.voc import parse_single_annot
from tqdm import tqdm

# Directorio de trabajo
DIR = "./"

# Archivo de configuración
config_path = DIR + "config.json"
with open(config_path) as config_buffer:
    config = json.loads(config_buffer.read())

#############################
# FUNCIONES AUXILIARES
#############################

def norm(im):
    """Normaliza una imagen de números reales a [0,1]"""

    return cv2.normalize(im, None, 0.0, 1.0, cv2.NORM_MINMAX)

def read_im(filename, color_flag = 1):
    """Devuelve una imagen de números reales adecuadamente leída en grises o en color.
        - filename: ruta de la imagen.
        - color_flag: indica si es en color (1) o en grises (0)."""

    try:
        im = cv2.imread(filename, color_flag)
        if len(im.shape) == 3:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    except:
        print("Error: no se ha podido cargar la imagen " + filename)
        quit()

    return im.astype(np.double)

def print_im(im, title = "", show = True, tam = (10, 10)):
    """Muestra una imagen cualquiera normalizada.
        - im: imagen a mostrar.
        - show: indica si queremos mostrar la imagen inmediatamente.
        - tam = (width, height): tamaño del plot."""

    show_title = len(title) > 0

    if show:
        fig = plt.figure(figsize = tam)

    im = norm(im)  # Normalizamos a [0,1]
    plt.imshow(im, interpolation = None, cmap = 'gray')
    plt.xticks([]), plt.yticks([])

    if show:
        if show_title:
            plt.title(title)
        plt.show()

def print_multiple_im(vim, titles = "", ncols = 2, tam = (10, 10)):
    """Muestra una sucesión de imágenes en la misma ventana, eventualmente con sus títulos.
        - vim: sucesión de imágenes a mostrar.
        - titles: o bien vacío o bien una sucesión de títulos del mismo tamaño que vim.
        - ncols: número de columnas del multiplot.
        - tam = (width, height): tamaño del multiplot."""

    show_title = len(titles) > 0

    nrows = len(vim) // ncols + (0 if len(vim) % ncols == 0 else 1)
    plt.figure(figsize = tam)

    for i in range(len(vim)):
        plt.subplot(nrows, ncols, i + 1)
        if show_title:
            plt.title(titles[i])
        print_im(vim[i], title = "", show = False)

    plt.show()

################################
# FUNCIONES DE DETECCIÓN
################################

def _detect_one(
    model,
    filein,
    img_dir,
    is_annot = False,
    show = True,
    show_ground_truth = False):
    if is_annot:
        # Analizamos la anotación
        im_inst = parse_single_annot(filein, img_dir)
        filename = im_inst['filename']
    else:
        filename = filein

    # Leemos la imagen
    im = read_im(filename)

    # Ground truth
    if show_ground_truth:
        ground_boxes = [BoundBox(obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'])
                           for obj in im_inst['object']]

    # Predecimos las bounding boxes
    pred_boxes = get_yolo_boxes(
        model,
        images = [im],
        net_h = config['model']['input_size'],
        net_w = config['model']['input_size'],
        anchors = config['model']['anchors'],
        obj_thresh = 0.5,
        nms_thresh = 0.45)[0]

    # Dibujamos las bounding boxes
    im_boxes = draw_boxes(im, pred_boxes, color = (0, 255, 25))

    if show_ground_truth:
        im_boxes = draw_boxes(im, ground_boxes, show_score = False, color = (255, 0, 25))

    # Mostramos la imagen
    if show:
        print_im(im_boxes)

    return im_boxes

def _detect_video(model, filein, fileout):
    # Abrimos el vídeo
    video_reader = cv2.VideoCapture(filein)

    # Recolectamos información del vídeo
    frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = video_reader.get(cv2.CAP_PROP_FPS)

    # Creamos vídeo de salida
    video_writer = cv2.VideoWriter(
        fileout,
        cv2.VideoWriter_fourcc(*'MPEG'),
        fps,
        (frame_w, frame_h))

    batch_size  = 1
    images      = []
    for i in tqdm(range(frames)):
        # Leemos frames del vídeo
        _, image = video_reader.read()
        images += [image]

        if (i%batch_size == 0) or (i == (frames-1) and len(images) > 0):
            # Predecimos bounding boxes
            batch_boxes = get_yolo_boxes(
                model,
                images,
                config['model']['input_size'],
                config['model']['input_size'],
                config['model']['anchors'],
                obj_thresh = 0.6,
                nms_thresh = 0.4)

            for i in range(len(images)):
                # Dibujamos bounding boxes
                draw_boxes(images[i], batch_boxes[i], show_score = False)

                # Escribimos imágenes en el vídeo de salida
                video_writer.write(images[i])

        images = []

    # Liberamos recursos
    video_reader.release()
    video_writer.release()

def main():
    # Modelo
    yolo = load_model(config['valid']['valid_model'])

    # Definimos una imagen de prueba
    annot = config['valid']['valid_annot_folder'] + "0_Parade_marchingband_1_404.xml"

    # Detectamos caras en la imagen de prueba
    _detect_one(yolo, annot, config['valid']['valid_image_folder'], show=True, is_annot = True, show_ground_truth = False)

    # Detectamos caras en vídeo
    #_detect_video(yolo, "widerface/test.mp4", "widerface/test_out.mp4", show_window = True)

if __name__ == "__main__":
    main()
