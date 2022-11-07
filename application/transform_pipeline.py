# We need to prepare image for processing,
# so fun will return bytes (????)
import io
import json
import torchvision.transforms as transform
from torchvision import models
from PIL import Image

model = models.densenet121(pretrained=True)
model.eval()
imagenet_class_index = json.load(open('application/static/imagenet_class_index.json'))


def transform_image(image_bytes):
    my_transform = transform.Compose([
        transform.Resize(255),
        transform.CenterCrop(224),
        transform.ToTensor(),
        transform.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transform(image).unsqueeze(0)


def get_prediction(image_bytes):
    tensor = transform_image(image_bytes)
    outputs = model.forward(tensor)
    # y_hat will contain the index of the predicted class id
    _, y_hat = outputs.max(1)
    predicted_idx = str(y_hat.item())
    return imagenet_class_index[predicted_idx]
