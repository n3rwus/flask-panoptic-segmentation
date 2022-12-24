import torch
import math
import matplotlib.pyplot as plt
from image_segmentation_transform_pipeline import transform_image, get_model

model = get_model()


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes)

    if torch.cuda.is_available():
        tensor = tensor.to(torch.device("cuda:0"))

    return model(tensor)


# Plot all the remaining masks
def print_remaining_masks():
    # compute the scores, excluding the "no-object" class (the last one)
    scores = torch.Tensor.cpu(
        out["pred_logits"]).softmax(-1)[..., :-1].max(-1)[0]
    # threshold the confidence
    keep = scores > 0.85

    # Plot all the remaining masks
    ncols = 5
    fig, axs = plt.subplots(ncols=ncols, nrows=math.ceil(
        keep.sum().item() / ncols), figsize=(18, 10))
    for line in axs:
        for a in line:
            a.axis('off')
    for i, mask in enumerate(torch.Tensor.cpu(out["pred_masks"])[keep]):
        ax = axs[i // ncols, i % ncols]
        ax.imshow(mask, cmap="cividis")
        ax.axis('off')
    return fig.tight_layout()
