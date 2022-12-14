import json

import torch.cuda

from application.transform_pipeline import get_model, transform_image

model = get_model()
imagenet_class_index = json.load(open('application/static/imagenet_class_index.json'))


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes)

    if torch.cuda.is_available():
        tensor = tensor.to(torch.device("cuda:0"))

    outputs = model.forward(tensor)
    # y_hat will contain the index of the predicted class id
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]
