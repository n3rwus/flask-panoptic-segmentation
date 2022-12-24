import torch
import math
import matplotlib.pyplot as plt
from image_segmentation_transform_pipeline import transform_image, get_model, postprocessor, get_tensor
import itertools
import seaborn as sns

palette = itertools.cycle(sns.color_palette())
model = get_model()
tensor = get_tensor()
out = model(tensor)


def print_panoptic_segmentation():
    return postprocessor(out, torch.as_tensor(
        tensor.shape[-2:]).unsqueeze(0))[0]


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
