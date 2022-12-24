import IPython.display
from PIL import Image
import requests
import io
import math
import matplotlib.pyplot as plt
from read_classes_from_file import read_coco_classes_from_file
import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as transform
import numpy


torch.set_grad_enabled(False)

CLASSES = read_coco_classes_from_file
model, postprocessor = torch.hub.load('facebookresearch/detr', 'detr_resnet101_panoptic', pretrained=True,
                                      return_postprocessor=True, num_classes=250)


def get_model():
    return model.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')).eval()


# Detectron2 uses a different numbering scheme, we build a conversion table
def conversion_table_for_Detectron2(CLASSES):
    coco2d2 = {}
    count = 0
    for i, c in enumerate(CLASSES):
        if c != "N/A":
            coco2d2[i] = count
            count += 1
    return coco2d2


def transform_image(image_bytes):
    transform = transform.Compose([
        transform.Resize(800),
        transform.ToTensor(),
        transform.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return transform(image).unsqueeze(0)
