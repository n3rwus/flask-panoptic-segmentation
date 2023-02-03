import torch.cuda
from application.detr_model.transform_pipeline import get_detr,  transform_image_for_segmentation

torch.set_grad_enabled(False)


detr, postprocessor = get_detr()
detr.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')).eval()


def get_segmentation(image_bytes):
    tensor = transform_image_for_segmentation(image_bytes)

    if torch.cuda.is_available():
        tensor = tensor.to(torch.device("cuda:0"))
    out = detr(tensor)
    return out, postprocessor(out, torch.as_tensor(tensor.shape[-2:]).unsqueeze(0))[0]
