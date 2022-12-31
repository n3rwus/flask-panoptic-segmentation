import base64
import io
import cv2
import math
import numpy
import torch
import itertools
from PIL import Image
import seaborn as sns
from copy import deepcopy
import matplotlib.pyplot as plt
from detectron2.data import MetadataCatalog
from panopticapi.utils import rgb2id
from detectron2.utils.visualizer import Visualizer


# compute the scores, excluding the "no-object" class (the last one)
def print_remaining_masks(out):
    # compute the scores, excluding the "no-object" class (the last one)
    scores = torch.Tensor.cpu(
        out["pred_logits"]).softmax(-1)[..., :-1].max(-1)[0]
    # threshold the confidence
    keep = scores > 0.85

    # Plot all the remaining masks
    ncols = 5
    buf = io.BytesIO()
    fig, axs = plt.subplots(ncols=ncols, nrows=math.ceil(
        keep.sum().item() / ncols), figsize=(18, 10))
    for line in axs:
        for a in line:
            a.axis('off')
    for i, mask in enumerate(torch.Tensor.cpu(out["pred_masks"])[keep]):
        ax = axs[i // ncols, i % ncols]
        ax.imshow(mask, cmap="cividis")
        ax.set_title(str(i))
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return data


# result = postprocessor(out, torch.as_tensor(tensor.shape[-2:]).unsqueeze(0))[0]
def print_panoptic_segmentation(result):
    buf = io.BytesIO()
    palette = itertools.cycle(sns.color_palette())

    # The segmentation is stored in a special-format png
    panoptic_seg = Image.open(io.BytesIO(result['png_string']))
    panoptic_seg = numpy.array(panoptic_seg, dtype=numpy.uint8).copy()
    # We retrieve the ids corresponding to each mask
    panoptic_seg_id = rgb2id(panoptic_seg)

    # Finally we color each mask individually
    panoptic_seg[:, :, :] = 0
    for ID in range(panoptic_seg_id.max() + 1):
        panoptic_seg[panoptic_seg_id == ID] = numpy.asarray(
            next(palette)) * 255
    plt.figure(figsize=(15, 15))
    plt.imshow(panoptic_seg)
    plt.axis('on')
    plt.savefig(buf, format="png", bbox_inches='tight')

    buf.seek(0)
    data = base64.b64encode(buf.read()).decode("ascii")
    return data


def print_detectron2_visualization(result, im):
    # We extract the segments info and the panoptic result from DETR's prediction
    segments_info = deepcopy(result["segments_info"])
    # Panoptic predictions are stored in a special format png
    panoptic_seg = Image.open(io.BytesIO(result['png_string']))
    final_w, final_h = panoptic_seg.size
    # We convert the png into an segment id map
    panoptic_seg = numpy.array(panoptic_seg, dtype=numpy.uint8)
    panoptic_seg = torch.from_numpy(rgb2id(panoptic_seg))

    # Detectron2 uses a different numbering of coco classes, here we convert the class ids accordingly
    meta = MetadataCatalog.get("coco_2017_val_panoptic_separated")
    for i in range(len(segments_info)):
        c = segments_info[i]["category_id"]
        segments_info[i]["category_id"] = meta.thing_dataset_id_to_contiguous_id[
            c] if segments_info[i]["isthing"] else meta.stuff_dataset_id_to_contiguous_id[c]

    # Finally we visualize the prediction
    v = Visualizer(numpy.array(im.copy().resize((final_w, final_h)))
                   [:, :, ::-1], meta, scale=1.0)
    v._default_font_size = 20
    v = v.draw_panoptic_seg_predictions(
        panoptic_seg, segments_info, area_threshold=0)
    np_image_array = Image.fromarray(
        cv2.cvtColor(v.get_image(), cv2.COLOR_BGR2RGB))
    buff = io.BytesIO()
    np_image_array.save(buff, format="png")
    data = base64.b64encode(buff.getvalue()).decode("utf-8")
    return data
