import io

import torch.cuda
import torchvision.transforms as transform
from PIL import Image


def get_detr():
    return torch.hub.load('facebookresearch/detr', 'detr_resnet101_panoptic', pretrained=True, return_postprocessor=True, num_classes=250)


def transform_image_for_segmentation(image_bytes):
    my_transform = transform.Compose([
        transform.Resize(800),
        transform.ToTensor(),
        transform.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return my_transform(image).unsqueeze(0)
