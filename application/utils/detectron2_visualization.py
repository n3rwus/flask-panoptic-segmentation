from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from PIL import Image
from copy import deepcopy
import cv2
from image_segmentation_inference import get_postprocessor_result
import io
import numpy
import torch
from panopticapi.utils import rgb2id
import IPython.display


def print_detectron2_visualization():
    result = get_postprocessor_result()
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
    return IPython.display(Image.fromarray(cv2.cvtColor(v.get_image(), cv2.COLOR_BGR2RGB)))
