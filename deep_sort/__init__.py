from .deep_sort import DeepSort
import yolo
from yolo import YOLO


__all__ = ['DeepSort', 'build_tracker','build_detector']


def build_tracker(use_cuda):
    return DeepSort(r'deep_sort/deep/checkpoint/ckpt.t7',# namesfile=cfg.DEEPSORT.CLASS_NAMES,
                max_dist=0.2, min_confidence=0.1,
                nms_max_overlap=0.5, max_iou_distance=0.7,
                max_age=70, n_init=3, nn_budget=100, use_cuda=True)

def build_detector():
    return YOLO()
    









