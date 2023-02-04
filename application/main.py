import io
import torch.cuda
from PIL import Image
from flask import Blueprint, render_template, request, redirect
from flask_login import login_required, current_user
from application.detr_model.inference import get_segmentation
from application.detr_model.image_segmentation import print_detectron2_visualization, print_panoptic_segmentation, print_remaining_masks


main = Blueprint('main', __name__)


@main.route('/')
def index():
    return render_template('index.html')


@main.route('/about')
def about():
    return render_template('about.html')


@main.route('/profile')
@login_required
def profile():
    return render_template('profile.html', name=current_user.name)


@main.route('/segmentation', methods=['GET', 'POST'])
@login_required
def segmentation():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return redirect(request.url)
        img_bytes = file.read()
        out, result = get_segmentation(image_bytes=img_bytes)
        data2 = print_panoptic_segmentation(result)
        data3 = print_detectron2_visualization(
            result, im=Image.open(io.BytesIO(img_bytes)))
        return render_template('result.html', panoptic=data2, detectron=data3, is_cuda_used="👍" if torch.cuda.is_available() else "👎")
    return render_template('index.html')
