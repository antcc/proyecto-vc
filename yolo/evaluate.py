import numpy as np
from .utils.utils import get_yolo_boxes

def evaluate_coco(model,
                 generator,
                 iou_start = 0.5,
                 iou_step = 0.05,
                 num_iou = 10,
                 obj_thresh=0.5,
                 nms_thresh=0.5,
                 net_h=416,
                 net_w=416,
                 save_path=""):
    # Avergage AP overmany  IoU thresholds
    iou_thresh_lst = np.array([iou_start + i * iou_step for i in range(num_iou)])

    # gather all detections and annotations
    all_detections     = [[None for i in range(generator.num_classes())] for j in range(generator.size())]
    all_annotations    = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

    for i in range(generator.size()):
        raw_image = [generator.load_image(i)]

        # make the boxes and the labels
        pred_boxes = get_yolo_boxes(model, raw_image, net_h, net_w, generator.get_anchors(), obj_thresh, nms_thresh)[0]

        score = np.array([box.get_score() for box in pred_boxes])
        pred_labels = np.array([box.label for box in pred_boxes])

        if len(pred_boxes) > 0:
            pred_boxes = np.array([[box.xmin, box.ymin, box.xmax, box.ymax, box.get_score()] for box in pred_boxes])
        else:
            pred_boxes = np.array([[]])

        # sort the boxes and the labels according to scores
        score_sort = np.argsort(-score)
        pred_labels = pred_labels[score_sort]
        pred_boxes  = pred_boxes[score_sort]

        # copy detections to all_detections
        for label in range(generator.num_classes()):
            all_detections[i][label] = pred_boxes[pred_labels == label, :]

        annotations = generator.load_annotation(i)

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()

    # compute mAP by comparing all detections and all annotations
    average_precisions = {}

    # Open file for output
    save = len(save_path) > 0
    f = None
    if save:
        f = open(save_path, "w")

    for label in range(generator.num_classes()):
        false_positives = [np.zeros((0,)) for j in range(num_iou)]
        true_positives  = [np.zeros((0,)) for j in range(num_iou)]
        scores          = np.zeros((0,))
        num_annotations = 0.0

        for i in range(generator.size()):
            detections           = all_detections[i][label]
            annotations          = all_annotations[i][label]
            num_annotations     += annotations.shape[0]
            detected_annotations = []

            if save:
                f.write("# " + generator.img_filename(i) + "\n")

            for d in detections:
                scores = np.append(scores, d[4])

                if save:
                    face = '{:.1f} {:.1f} {:.1f} {:.1f} {:f}\n'.format(d[0], d[1], d[2] - d[0], d[3] - d[1], d[4])
                    f.write(face)

                if annotations.shape[0] == 0:
                    for j in range(num_iou):
                        false_positives[j] = np.append(false_positives[j], 1)
                        true_positives[j]  = np.append(true_positives[j], 0)
                    continue

                overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap         = overlaps[0, assigned_annotation]

                if assigned_annotation in detected_annotations:
                    for j in range(num_iou):
                        false_positives[j] = np.append(false_positives[j], 1)
                        true_positives[j]  = np.append(true_positives[j], 0)
                else:
                    for j, iou_thresh in enumerate(iou_thresh_lst):
                        if max_overlap >= iou_thresh:
                            false_positives[j] = np.append(false_positives[j], 0)
                            true_positives[j]  = np.append(true_positives[j], 1)
                        else:
                            false_positives[j] = np.append(false_positives[j], 1)
                            true_positives[j]  = np.append(true_positives[j], 0)
                    if (max_overlap >= iou_thresh_lst).any():
                        detected_annotations.append(assigned_annotation)

        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0
            continue

        # sort by score
        indices = np.argsort(-scores)
        recall = [np.zeros((0,)) for j in range(num_iou)]
        precision = [np.zeros((0,)) for j in range(num_iou)]
        average_precision = 0.0
        for j in range(num_iou):
            false_positives[j] = false_positives[j][indices]
            true_positives[j]  = true_positives[j][indices]

            # compute false positives and true positives
            false_positives[j] = np.cumsum(false_positives[j])
            true_positives[j]  = np.cumsum(true_positives[j])

            # compute recall and precision
            recall[j]    = true_positives[j] / num_annotations
            precision[j] = true_positives[j] / np.maximum(true_positives[j] + false_positives[j], np.finfo(np.float64).eps)

            # compute average precision
            average_precision = average_precision + compute_ap(recall[j], precision[j])

        average_precisions[label] = average_precision / float(num_iou)

    return average_precisions

def compute_overlap(a, b):
    """
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap
