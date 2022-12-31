import json
import torch.cuda
from application.transform_pipeline import get_densenet121, get_detr, transform_image, transform_image_for_segmentation

torch.set_grad_enabled(False)


# densenet121 = get_densenet121()
detr, postprocessor = get_detr()
detr.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')).eval()


# imagenet_class_index = json.load(
#     open('application/static/imagenet_class_index.json'))


# def get_prediction(image_bytes):
#     tensor = transform_image(image_bytes)

#     if torch.cuda.is_available():
#         tensor = tensor.to(torch.device("cuda:0"))

#     outputs = densenet121.forward(tensor)
#     # y_hat will contain the index of the predicted class id
#     _, y_hat = outputs.max(1)
#     predicted_idx = str(y_hat.item())
#     return imagenet_class_index[predicted_idx]


def get_segmentation(image_bytes):
    tensor = transform_image_for_segmentation(image_bytes)

    if torch.cuda.is_available():
        tensor = tensor.to(torch.device("cuda:0"))
    out = detr(tensor)
    return out, postprocessor(out, torch.as_tensor(tensor.shape[-2:]).unsqueeze(0))[0]
