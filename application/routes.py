import base64
import torch.cuda
from io import BytesIO
from matplotlib.figure import Figure

from application import app
from application.image_segmentation import transform_image
from flask import render_template, request, jsonify, redirect

from application.inference import get_prediction
from application.transform_pipeline import format_class_name


@app.route('/about')
def about():
    return render_template("about.html", about=True)


@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
@app.route('/home', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return redirect(request.url)
        img_bytes = file.read()
        class_id, class_name = get_prediction(image_bytes=img_bytes)
        class_name = format_class_name(class_name)
        return render_template('result.html', class_id=class_id,
                               class_name=class_name, image_after=img_bytes, is_cuda_used=torch.cuda.is_available())
    return render_template('index.html')


@app.route('/test', methods=['GET', 'POST'])
def test():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return redirect(request.url)
        img_bytes = file.read()
        out = transform_image(image_bytes=img_bytes)
        
        return render_template('result.html')
    return render_template('index.html')


@app.route("/hello")
def hello():
    # Generate the figure **without using pyplot**.
    fig = Figure()
    ax = fig.subplots()
    ax.plot([1, 2])
    # Save it to a temporary buffer.
    buf = BytesIO()
    fig.savefig(buf, format="png")
    # Embed the result in the html output.
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return f"<img src='data:image/png;base64,{data}'/>"
