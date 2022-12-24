import IPython.display
from PIL import Image
import requests
import io
import math
import matplotlib.pyplot as plt

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as transform
import numpy

torch.set_grad_enabled(False)


model, postprocessor = torch.hub.load('facebookresearch/detr', 'detr_resnet101_panoptic', pretrained=True,
                                      return_postprocessor=True, num_classes=250)
