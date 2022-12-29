import io
import math
import numpy
import torch
import requests
import itertools
import panopticapi
from torch import nn
from PIL import Image
import seaborn as sns
import IPython.display
import matplotlib.pyplot as plt
from torchvision.models import resnet50
import torchvision.transforms as transform
from panopticapi.utils import id2rgb, rgb2id

torch.set_grad_enabled(False)

model, postprocessor = torch.hub.load('facebookresearch/detr', 'detr_resnet101_panoptic', pretrained=True,
                                      return_postprocessor=True, num_classes=250)


def get_model():
    return model.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')).eval()


def read_coco_classes_from_file(path='application/static/coco_classes.txt'):
    coco_classes_file = open(path, "r")

    coco_classes = coco_classes_file.read()
    coco_classes_file.close()
    return coco_classes.split(",")


# Detectron2 uses a different numbering scheme, we build a conversion table
def conversion_table(CLASSES):
    coco2d2 = {}
    count = 0
    for i, c in enumerate(CLASSES):
        if c != "N/A":
            coco2d2[i] = count
            count += 1


# standard PyTorch mean-std input image normalization
transform = transform.Compose([
    transform.Resize(800),
    transform.ToTensor(),
    transform.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def transform_image(image_bytes):
    transform = transform.Compose([
        transform.Resize(800),
        transform.ToTensor(),
        transform.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225])
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return transform(image).unsqueeze(0)
