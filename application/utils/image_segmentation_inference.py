
import torch
from image_segmentation_transform_pipeline import transform_image, get_model

model = get_model()


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes)

    if torch.cuda.is_available():
        tensor = tensor.to(torch.device("cuda:0"))

    return model(tensor)
