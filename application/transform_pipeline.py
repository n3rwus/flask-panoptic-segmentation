import io

import torch.cuda
import torchvision.transforms as transform
from torchvision import models
from PIL import Image


def get_model(path=''):
    return models.densenet121(pretrained=True).to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')).eval()


def transform_image(image_bytes):
    my_transform = transform.Compose([
        transform.Resize(255),
        transform.CenterCrop(224),
        transform.ToTensor(),
        transform.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    return my_transform(image).unsqueeze(0)


def format_class_name(class_name):
    class_name = class_name.replace('_', ' ')
    class_name = class_name.title()
    return class_name
