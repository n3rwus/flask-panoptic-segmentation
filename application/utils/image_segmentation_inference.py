import torch
import math
import matplotlib.pyplot as plt
from image_segmentation_transform_pipeline import transform_image, get_model, postprocessor, get_tensor
import itertools
import seaborn as sns
import numpy
import io
from PIL import Image
import panopticapi
from panopticapi.utils import id2rgb, rgb2id


palette = itertools.cycle(sns.color_palette())
model = get_model()
tensor = get_tensor()
out = model(tensor)


def get_postprocessor_result():
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


def print_panoptic_segmentation():
    # The post-processor expects as input the target size of the predictions (which we set here to the image size)
    result = get_postprocessor_result()
    # The segmentation is stored in a special-format png
    panoptic_seg = Image.open(io.BytesIO(result['png_string']))
    panoptic_seg = numpy.array(panoptic_seg, dtype=numpy.uint8).copy()
    # Retrieve the ids corresponding to each mask
    panoptic_seg_id = rgb2id(panoptic_seg)
    # Color each mask individually
    panoptic_seg[:, :, :] = 0
    for ID in range(panoptic_seg_id.max() + 1):
        panoptic_seg[panoptic_seg_id == ID] = numpy.asarray(
            next(palette)) * 255
    plt.figure(figsize=(15, 15))
    plt.imshow(panoptic_seg)
    plt.axis('on')
    return plt.show()
