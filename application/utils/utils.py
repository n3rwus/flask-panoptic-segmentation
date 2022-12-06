import cv2
import numpy as np
import random
import torch

from coco_names import COCO_INSTANCE_CATEGORY_NAMES as coco_names

COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))
